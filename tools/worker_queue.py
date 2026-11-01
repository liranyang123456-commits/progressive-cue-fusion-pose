#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential benchmark queue for a GPU worker that received tools/make_worker_bundle.py output.

  python worker_queue.py --root %USERPROFILE%/pose_bench --python %USERPROFILE%/venvs/pose/Scripts/python.exe \
         --jobs bunny:grid_adam:100,bunny:adam_dr:100,...

Each job runs run_benchmark_compare.py (resumable: finished GT views are skipped) with the strict no-GT
protocol and identical cached observations. Progress is appended to <root>/worker_queue_log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CORE = ("grid_adam", "rand_restart_adam", "adam_dr")  # the baselines discussed individually by the reviewers
REST = ("hybrid_dr", "rand_topk_only", "greedy_bt_local", "grid_adam_top16")
DEFAULT_JOBS = [
    # pass 1: bunny / dragon, first 50 of the N=100 views, core baselines (ours-family runs on the control machine)
    *[f"{d}:{m}:100:50" for m in CORE for d in ("bunny", "dragon")],
    # pass 2: additional meshes, N=50, core baselines
    *[f"{d}:{m}:50" for m in CORE for d in ("armadillo", "happy", "lucy")],
    # pass 3: remaining baselines, bunny / dragon first 50
    *[f"{d}:{m}:100:50" for m in REST for d in ("bunny", "dragon")],
    # pass 4: remaining baselines, additional meshes
    *[f"{d}:{m}:50" for m in REST for d in ("armadillo", "happy", "lucy")],
    # pass 5: second half of the bunny / dragon views, all baselines
    *[f"{d}:{m}:100" for m in CORE + REST for d in ("bunny", "dragon")],
]


def finished_rows(out_dir: Path) -> int:
    p = out_dir / "pose_metrics_summary.csv"
    if not p.exists():
        return 0
    return max(0, len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]) - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--jobs", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    jobs = [j.strip() for j in args.jobs.split(",") if j.strip()] or DEFAULT_JOBS
    log = root / "worker_queue_log.jsonl"
    env = os.environ.copy()
    env["PIPELINE_PYTHON"] = args.python
    env["VAL_STRICT_NO_GT"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    for j in jobs:
        parts = j.split(":")
        ds, m, n = parts[0], parts[1], int(parts[2])
        k = int(parts[3]) if len(parts) > 3 else 0
        out_dir = root / "benchmark_runs" / f"{ds}_{n}gt" / m
        done = finished_rows(out_dir)
        if done >= n or (k and done >= k):
            print(f"[skip] {j}: {done}/{n} done", flush=True)
            continue
        cmd = [args.python, str(root / "run_benchmark_compare.py"), "--dataset", ds, "--num-gt", str(n), "--methods", m]
        if k:
            cmd += ["--max-views", str(max(0, k - done))]
        print(f"[start] {j} ({done}/{n} done) {datetime.now().strftime('%H:%M:%S')}", flush=True)
        if args.dry_run:
            print("   ", " ".join(cmd))
            continue
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(root), env=env).returncode
        rec = {"job": j, "rc": rc, "rows": finished_rows(out_dir), "elapsed_s": time.time() - t0, "finished_at": datetime.now(timezone.utc).isoformat()}
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[done] {j} rc={rc} rows={rec['rows']} {rec['elapsed_s']/60:.1f} min", flush=True)
    print("[queue] finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
