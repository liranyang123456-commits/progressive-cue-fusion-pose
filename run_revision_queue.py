#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revision benchmark queue (strict no-GT pipeline).

* Archives the legacy (GT-leaking) results once to benchmark_runs_legacy_leaky/ and
  copies only the cached ground-truth observations (gt_poses.csv, gt_XX/gt_refs_*.npz,
  gt_XX/*.png) back, so every method is re-run on identical observations.
* Executes (dataset, method, num_gt) jobs with K concurrent GPU workers; each job is a
  subprocess of run_benchmark_compare.run_method (same env/config path as before).
* Skips jobs whose pose_metrics_summary.csv is already complete.

Usage:
  python run_revision_queue.py --plan core --workers 3
  python run_revision_queue.py --jobs bunny:ours_v10:100,dragon:grid_adam:100 --workers 2
  python run_revision_queue.py --plan core --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_benchmark_compare import (  # noqa: E402
    DATASETS,
    GT_CACHE_METHOD,
    METHODS,
    PYTHON,
    _gt_cache_complete,
    _gt_tag,
    _run_is_complete,
)

LEGACY_ROOT = ROOT / "benchmark_runs_legacy_leaky"
RUNS_ROOT = ROOT / "benchmark_runs"
QUEUE_LOG = ROOT / "benchmark_results" / "revision_queue_log.jsonl"

# Core method matrix for the revision (GPU budget: ours-family runs cost ~8-11 min/view).
# Geometry-only selection (RGB ablation) is derived offline from the ours_v10 artefacts
# (identical beam candidates, geometric selector) by tools/derive_geom_only.py -> no extra run.
CORE_MAIN_100 = [  # bunny / dragon, N=100 (headline + cheap baselines/ablations)
    "ours_v10", "greedy_bt_local", "rand_topk_only",
    "grid_adam", "grid_adam_top16", "rand_restart_adam", "adam_dr", "hybrid_dr",
]
CORE_MAIN_50 = ["ours_v10_noda"]                 # bunny / dragon, first 50 views (DA control)
CORE_MAIN_30 = ["ours_v10_fixedmr", "ours_v10_1shell"]  # bunny / dragon, first 30 views (radius schedule)
CORE_NEW_50 = ["ours_v10", "greedy_bt_local", "rand_topk_only", "grid_adam", "grid_adam_top16",
               "rand_restart_adam", "adam_dr"]   # armadillo / happy / lucy, N=50
CORE_NEW_30 = ["ours_v10_noda"]                  # armadillo / happy / lucy, first 30 views
# Heavy jobs first so that the tail of the schedule is filled with cheap baselines.
HEAVY = {"ours_v10", "ours_v10_noda", "ours_v10_geom", "ours_v10_fixedmr", "ours_v10_1shell"}


def build_plan(name: str) -> list[tuple[str, str, int]]:
    """Ordered job list. Priority: headline meshes (bunny, dragon) complete first
    (GT cache -> ours -> cheap baselines -> DA control -> radius ablations), then the three
    additional meshes in the same order. The queue keeps this order (no re-sorting)."""
    jobs: list[tuple[str, str, int]] = []
    cheap_main = [m for m in CORE_MAIN_100 if m != GT_CACHE_METHOD]
    cheap_new = [m for m in CORE_NEW_50 if m != GT_CACHE_METHOD]
    if name in ("core", "main"):
        for ds in ("bunny", "dragon"):
            jobs.append((ds, GT_CACHE_METHOD, 100))
        for ds in ("bunny", "dragon"):
            jobs += [(ds, m, 100) for m in cheap_main]
        for ds in ("bunny", "dragon"):
            jobs += [(ds, m, 50) for m in CORE_MAIN_50]
        for ds in ("bunny", "dragon"):
            jobs += [(ds, m, 30) for m in CORE_MAIN_30]
    if name == "ours_rest":
        # Ours-family jobs only (the cheap baselines run on the GPU worker, see tools/worker_queue.py);
        # the bunny/dragon ours_v10 N=100 runs are assumed to be running/finished separately.
        for ds in ("bunny", "dragon"):
            jobs += [(ds, m, 50) for m in CORE_MAIN_50]
        for ds in ("armadillo", "happy", "lucy"):
            jobs.append((ds, GT_CACHE_METHOD, 50))
        for ds in ("bunny", "dragon"):
            jobs += [(ds, m, 30) for m in CORE_MAIN_30]
        for ds in ("armadillo", "happy", "lucy"):
            jobs += [(ds, m, 30) for m in CORE_NEW_30]
        return jobs
    if name in ("core", "new"):
        for ds in ("armadillo", "happy", "lucy"):
            jobs.append((ds, GT_CACHE_METHOD, 50))
        for ds in ("armadillo", "happy", "lucy"):
            jobs += [(ds, m, 50) for m in cheap_new]
        for ds in ("armadillo", "happy", "lucy"):
            jobs += [(ds, m, 30) for m in CORE_NEW_30]
    return jobs


def archive_legacy_once() -> None:
    """Move the leaky results aside, keep only GT observation caches in benchmark_runs/."""
    marker = RUNS_ROOT / ".revision_clean"
    if marker.exists():
        return
    if not RUNS_ROOT.exists():
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        return
    LEGACY_ROOT.mkdir(parents=True, exist_ok=True)
    for run_root in sorted(RUNS_ROOT.glob("*gt")):
        dst = LEGACY_ROOT / run_root.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(run_root), str(dst))
        print(f"[archive] {run_root.name} -> {dst}")
        # Re-create the run root with only the GT cache of the reference method.
        cache_src = dst / GT_CACHE_METHOD
        if cache_src.is_dir() and (cache_src / "gt_poses.csv").exists():
            cache_dst = run_root / GT_CACHE_METHOD
            cache_dst.mkdir(parents=True, exist_ok=True)
            for fname in ("gt_poses.csv", "gt_poses_pose4x4.npy"):
                if (cache_src / fname).exists():
                    shutil.copy2(cache_src / fname, cache_dst / fname)
            n = 0
            for tag_dir in sorted(cache_src.glob("gt_*")):
                if not tag_dir.is_dir():
                    continue
                out = cache_dst / tag_dir.name
                out.mkdir(exist_ok=True)
                for f in tag_dir.iterdir():
                    if f.name in ("gt_refs_full.npz", "gt_refs_search.npz", "gt_rgb_ref.npy",
                                  "gt_camera_spherical.txt") or f.name.startswith(("gt_", "target_")) and f.suffix == ".png":
                        shutil.copy2(f, out / f.name)
                n += 1
            print(f"[gt-cache] restored {n} views for {run_root.name}/{GT_CACHE_METHOD}")
    marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")


def derive_cache_from_larger(ds: str, num_gt: int) -> bool:
    """Build <ds>_<N>gt/<GT_CACHE_METHOD>/ from a complete larger cache of the same dataset.
    GT poses are generated by a seeded sequence, so the first N views of an N'>N run are
    identical to an N-view run; copying the cached observations keeps them bit-identical."""
    dst = RUNS_ROOT / f"{ds}_{num_gt}gt" / GT_CACHE_METHOD
    if _gt_cache_complete(dst, num_gt):
        return True
    for src_root in sorted(RUNS_ROOT.glob(f"{ds}_*gt"), key=lambda q: -int(q.name.split('_')[-1][:-2])):
        n_src = int(src_root.name.split("_")[-1][:-2])
        src = src_root / GT_CACHE_METHOD
        if n_src <= num_gt or not _gt_cache_complete(src, n_src):
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for i in range(num_gt):
            tag = _gt_tag(i)
            (dst / tag).mkdir(exist_ok=True)
            for f in (src / tag).iterdir():
                if f.name in ("gt_refs_full.npz", "gt_refs_search.npz", "gt_rgb_ref.npy", "gt_camera_spherical.txt"):
                    shutil.copy2(f, dst / tag / f.name)
        # gt_poses.csv is rewritten by the main script; a truncated copy keeps the cache check happy
        lines = (src / "gt_poses.csv").read_text(encoding="utf-8").splitlines()
        (dst / "gt_poses.csv").write_text("\n".join(lines[: num_gt + 1]) + "\n", encoding="utf-8")
        print(f"[gt-cache] derived {ds} N={num_gt} from N={n_src}")
        return _gt_cache_complete(dst, num_gt)
    return False


def _log(rec: dict) -> None:
    QUEUE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_job(ds: str, method: str, num_gt: int, worker: int, dry: bool) -> int:
    out_dir = RUNS_ROOT / f"{ds}_{num_gt}gt" / method
    if _run_is_complete(out_dir, num_gt):
        print(f"[w{worker}] skip {ds}/{method} N={num_gt}: complete")
        return 0
    cmd = [PYTHON, str(ROOT / "run_benchmark_compare.py"), "--dataset", ds, "--num-gt", str(num_gt),
           "--methods", method, "--record"]
    print(f"[w{worker}] start {ds}/{method} N={num_gt}  {datetime.now().strftime('%H:%M:%S')}")
    if dry:
        print("   ", " ".join(cmd))
        return 0
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["VAL_STRICT_NO_GT"] = "1"
    rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
    dt = time.time() - t0
    ok = rc == 0 and _run_is_complete(out_dir, num_gt)
    _log({"dataset": ds, "method": method, "num_gt": num_gt, "rc": rc, "ok": ok, "elapsed_s": dt,
          "finished_at": datetime.now(timezone.utc).isoformat()})
    print(f"[w{worker}] done  {ds}/{method} N={num_gt} ok={ok} {dt/60:.1f} min")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="", help="core | main | new | ours_rest")
    ap.add_argument("--jobs", default="", help="comma list dataset:method:num_gt")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-archive", action="store_true")
    args = ap.parse_args()

    jobs: list[tuple[str, str, int]] = []
    if args.plan:
        jobs += build_plan(args.plan)
    for j in [x for x in args.jobs.split(",") if x.strip()]:
        ds, m, n = j.split(":")
        jobs.append((ds, m, int(n)))
    jobs = [j for j in jobs if j[0] in DATASETS and j[1] in METHODS and Path(DATASETS[j[0]]["model"]).exists()]
    if not jobs:
        print("no runnable jobs")
        return 0
    if not args.no_archive and not args.dry_run:
        archive_legacy_once()

    # The GT-cache method must finish before other methods of the same dataset start
    # (they copy its gt_XX caches). Enforce with per-dataset events.
    ready: dict[str, threading.Event] = {}
    for ds, m, n in jobs:
        key = f"{ds}_{n}"
        ready.setdefault(key, threading.Event())
        if _gt_cache_complete(RUNS_ROOT / f"{key}gt" / GT_CACHE_METHOD, n):
            ready[key].set()
    q: Queue = Queue()
    for j in jobs:
        q.put(j)
    failures: list[tuple] = []
    lock = threading.Lock()

    def worker(wid: int) -> None:
        while True:
            try:
                ds, m, n = q.get_nowait()
            except Exception:
                return
            key = f"{ds}_{n}"
            if m != GT_CACHE_METHOD and not ready[key].is_set():
                # Wait for the GT cache of this dataset: produced by the GT-cache job of the same N,
                # or derived from a completed larger-N cache of the same dataset.
                while not ready[key].is_set():
                    if derive_cache_from_larger(ds, n):
                        ready[key].set()
                        break
                    if args.dry_run:
                        break
                    ready[key].wait(timeout=60.0)
            rc = run_job(ds, m, n, wid, args.dry_run)
            if m == GT_CACHE_METHOD or _gt_cache_complete(RUNS_ROOT / f"{key}gt" / GT_CACHE_METHOD, n):
                ready[key].set()
            if rc != 0:
                with lock:
                    failures.append((ds, m, n))
            q.task_done()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(max(1, args.workers))]
    for t in threads:
        t.start()
        time.sleep(2.0)
    for t in threads:
        t.join()

    if not args.dry_run:
        subprocess.run([PYTHON, str(ROOT / "tools" / "rebuild_master_jsonl.py")], cwd=str(ROOT), check=False)
        subprocess.run([PYTHON, str(ROOT / "record_benchmark_results.py"), "--report"], cwd=str(ROOT), check=False)
    print(f"[queue] finished; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
