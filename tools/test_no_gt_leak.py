#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground-truth leak test (CI gate for the pose-refinement benchmark).

Protocol
--------
For every method under test we run the benchmark twice on the SAME cached
observations (silhouette / depth / normals / RGB rendered once from GT set A):

  Run A : GT labels = set A (VAL_GT_POSE_SEED=<seed_a>), observations rendered and cached.
  Run B : GT labels = set B (VAL_GT_POSE_SEED=<seed_b>), observations COPIED from run A
          (VAL_PRECOMPUTE_GT=0), so only the ground-truth *labels* change.

If the pipeline never consults the ground truth during search / scoring /
selection, the estimated poses (est_dist, est_elev_deg, est_azim_deg) must be
identical between A and B, while the reported errors necessarily differ
(the labels differ). Any difference in the estimates is a GT leak.

The optional ``--legacy`` flag repeats the pair with VAL_STRICT_NO_GT=0 to
document how much the historical pipeline depended on the GT labels.

Usage
-----
  python tools/test_no_gt_leak.py --dataset bunny --methods ours_v10,grid_adam --num-gt 3
  python tools/test_no_gt_leak.py --dataset bunny --methods ours_v10 --legacy
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_benchmark_compare import DATASETS, MAIN, METHODS, PYTHON, V9_SHARED  # noqa: E402

EST_COLS = ("est_dist", "est_elev_deg", "est_azim_deg")
GT_COLS = ("gt_dist", "gt_elev_deg", "gt_azim_deg")
OBS_FILES = ("gt_refs_full.npz", "gt_refs_search.npz", "gt_rgb_ref.npy")


def _tags(num_gt: int) -> list[str]:
    return [f"gt_{i:02d}" for i in range(int(num_gt))]


def _base_env(dataset: str, method: str, num_gt: int, out_dir: Path, strict: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VAL_GT_TAGS", None)
    env["BENCHMARK_PRESERVE_ENV"] = "1"
    env["FORCE_CODE_RUN_PARAMS"] = "1"
    env.update(V9_SHARED)
    env.update(DATASETS[dataset].get("extra", {}))
    env.update(METHODS[method])
    model = DATASETS[dataset]["model"]
    portable = ROOT / "benchmark_assets" / f"{dataset}_portable.ply"
    if not Path(model).exists() and portable.exists():
        # GPU worker without the original mesh tree: use the exact mesh exported by the reference run
        model = str(portable)
        env["VAL_MESH_SIMPLIFY"] = "0"
        env.pop("VAL_MESH_TARGET_TRIS", None)
    env["INPUT_MODEL_PATH"] = model
    env["OUTPUT_DIR"] = str(out_dir)
    env["NUM_GT_POSES"] = str(num_gt)
    env["VAL_GT_TAGS"] = ",".join(_tags(num_gt))
    env["VAL_STRICT_NO_GT"] = "1" if strict else "0"
    if not strict:
        # Re-enable the historical GT-dependent switches to measure the legacy leak.
        env["VAL_LOCAL_BEAM_COND_USE_GT_ELEV"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run([PYTHON, str(MAIN)], cwd=str(ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def _read_metrics(out_dir: Path) -> dict[str, dict[str, float]]:
    p = out_dir / "pose_metrics_summary.csv"
    if not p.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tag = str(row.get("tag", "")).strip()
            if not tag:
                continue
            rows[tag] = {k: float(row[k]) for k in EST_COLS + GT_COLS if k in row and row[k] not in ("", "nan")}
    return rows


def _copy_observations(src: Path, dst: Path, num_gt: int) -> None:
    for tag in _tags(num_gt):
        (dst / tag).mkdir(parents=True, exist_ok=True)
        for fname in OBS_FILES:
            s = src / tag / fname
            if s.exists():
                shutil.copy2(s, dst / tag / fname)


def _max_abs_diff(a: dict[str, float], b: dict[str, float], cols: tuple[str, ...]) -> float:
    diffs = []
    for c in cols:
        if c in a and c in b:
            d = abs(a[c] - b[c])
            if c == "est_azim_deg" or c == "gt_azim_deg":
                d = min(d, 360.0 - d)
            diffs.append(d)
    return max(diffs) if diffs else float("nan")


def run_pair(
    *, dataset: str, method: str, num_gt: int, seed_a: int, seed_b: int, root: Path, strict: bool, force: bool, repeat: bool = False,
    repeats: int = 1
) -> dict:
    label = "strict" if strict else "legacy"
    pair_root = root / dataset / method / label
    dir_a, dir_b = pair_root / "A", pair_root / "B"
    if force:
        shutil.rmtree(pair_root, ignore_errors=True)
    pair_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if not (dir_a / "pose_metrics_summary.csv").exists():
        env_a = _base_env(dataset, method, num_gt, dir_a, strict)
        env_a["VAL_GT_POSE_SEED"] = str(seed_a)
        env_a["VAL_PRECOMPUTE_GT"] = "1"
        rc = _run(env_a, pair_root / "run_A.log")
        if rc != 0:
            return {"dataset": dataset, "method": method, "mode": label, "status": "run_A_failed", "rc": rc}
    if not (dir_b / "pose_metrics_summary.csv").exists():
        dir_b.mkdir(parents=True, exist_ok=True)
        _copy_observations(dir_a, dir_b, num_gt)
        env_b = _base_env(dataset, method, num_gt, dir_b, strict)
        env_b["VAL_GT_POSE_SEED"] = str(seed_b)
        env_b["VAL_PRECOMPUTE_GT"] = "0"
        rc = _run(env_b, pair_root / "run_B.log")
        if rc != 0:
            return {"dataset": dataset, "method": method, "mode": label, "status": "run_B_failed", "rc": rc}
    # control runs A2..A{k+1}: identical labels AND identical observations as A -> any difference is run-to-run
    # nondeterminism of the method itself (e.g. atomic adds in the rasteriser backward pass). Several repeats
    # bound that spread better than a single one.
    rep_dirs = [pair_root / f"A{i}" for i in range(2, 2 + max(1, repeats))] if repeat else []
    for dir_ai in rep_dirs:
        if (dir_ai / "pose_metrics_summary.csv").exists():
            continue
        dir_ai.mkdir(parents=True, exist_ok=True)
        _copy_observations(dir_a, dir_ai, num_gt)
        env_ai = _base_env(dataset, method, num_gt, dir_ai, strict)
        env_ai["VAL_GT_POSE_SEED"] = str(seed_a)
        env_ai["VAL_PRECOMPUTE_GT"] = "0"
        rc = _run(env_ai, pair_root / f"run_{dir_ai.name}.log")
        if rc != 0:
            return {"dataset": dataset, "method": method, "mode": label, "status": f"run_{dir_ai.name}_failed", "rc": rc}
    elapsed = time.time() - t0

    ma, mb = _read_metrics(dir_a), _read_metrics(dir_b)
    # all repeat runs that exist on disk (also from earlier invocations)
    rep_metrics = []
    for d in sorted(pair_root.glob("A[0-9]*")):
        if (d / "pose_metrics_summary.csv").exists():
            rep_metrics.append(_read_metrics(d))
    ma2 = rep_metrics[0] if rep_metrics else {}
    per_tag = []
    for tag in _tags(num_gt):
        if tag not in ma or tag not in mb:
            per_tag.append({"tag": tag, "status": "missing"})
            continue
        row = {
            "tag": tag,
            "est_diff_max": _max_abs_diff(ma[tag], mb[tag], EST_COLS),
            "gt_diff_max": _max_abs_diff(ma[tag], mb[tag], GT_COLS),
            "A": ma[tag],
            "B": mb[tag],
        }
        diffs = [_max_abs_diff(ma[tag], mr[tag], EST_COLS) for mr in rep_metrics if tag in mr]
        if diffs:
            row["repeat_diff_max"] = max(diffs)
            row["repeat_diffs"] = diffs
            row["A2"] = ma2[tag]
        per_tag.append(row)
    return {
        "dataset": dataset,
        "method": method,
        "mode": label,
        "status": "ok",
        "elapsed_sec": elapsed,
        "per_tag": per_tag,
        "has_repeat": bool(ma2),
        "n_repeats": len(rep_metrics),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Ground-truth leak test for the pose benchmark.")
    ap.add_argument("--dataset", default="bunny")
    ap.add_argument("--methods", default="ours_v10,grid_adam,rand_restart_adam")
    ap.add_argument("--num-gt", type=int, default=3)
    ap.add_argument("--seed-a", type=int, default=12291)
    ap.add_argument("--seed-b", type=int, default=777)
    ap.add_argument("--tol", type=float, default=1e-4, help="max allowed |est_A - est_B|")
    ap.add_argument("--root", default=str(ROOT / "leak_test_runs"))
    ap.add_argument("--legacy", action="store_true", help="also run the pair with VAL_STRICT_NO_GT=0")
    ap.add_argument("--repeat", action="store_true", help="also repeat run A with identical labels (nondeterminism control)")
    ap.add_argument("--repeats", type=int, default=1, help="number of identical-label repeat runs (A2..A{k+1}); implies --repeat")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if args.repeats > 1:
        args.repeat = True
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = []
    all_ok = True
    for m in methods:
        if m not in METHODS:
            print(f"[skip] unknown method {m}")
            continue
        modes = [True] + ([False] if args.legacy else [])
        for strict in modes:
            res = run_pair(
                dataset=args.dataset, method=m, num_gt=args.num_gt, seed_a=args.seed_a, seed_b=args.seed_b,
                root=root, strict=strict, force=args.force, repeat=args.repeat, repeats=args.repeats,
            )
            results.append(res)
            label = res["mode"]
            if res["status"] != "ok":
                print(f"[{label}] {m}: {res['status']} (rc={res.get('rc')})")
                if strict:
                    all_ok = False
                continue
            worst_est = max((t.get("est_diff_max", float('inf')) for t in res["per_tag"]), default=float("inf"))
            min_gt = min((t.get("gt_diff_max", 0.0) for t in res["per_tag"]), default=0.0)
            rep = [t["repeat_diff_max"] for t in res["per_tag"] if "repeat_diff_max" in t]
            worst_rep = max(rep) if rep else None
            if worst_est <= args.tol and min_gt > 1.0:
                verdict = "PASS (bitwise)"
            elif worst_rep is not None and worst_rep > args.tol and min_gt > 1.0:
                # the method is not bitwise reproducible even with identical labels: a label dependence can only
                # be claimed if the label-swap difference exceeds the repeat difference by a clear margin
                verdict = "PASS (within nondeterminism)" if worst_est <= 2.0 * worst_rep + args.tol else "FAIL"
            else:
                verdict = "FAIL"
            res["verdict"] = verdict
            res["worst_est_diff"] = worst_est
            res["worst_repeat_diff"] = worst_rep
            if strict and not verdict.startswith("PASS"):
                all_ok = False
            rep_txt = f"  max|est_A-est_A2|={worst_rep:.3e}" if worst_rep is not None else ""
            print(
                f"[{label}] {args.dataset}/{m}: max|est_A-est_B|={worst_est:.3e}{rep_txt}  "
                f"min|gt_A-gt_B|={min_gt:.2f}  -> {verdict if strict else 'documented leak' if worst_est > args.tol else 'no dependence'}"
            )
            for t in res["per_tag"]:
                if t.get("status") == "missing":
                    print(f"    {t['tag']}: missing")
                    continue
                a, b = t["A"], t["B"]
                print(
                    f"    {t['tag']}: est A=({a['est_dist']:.4f},{a['est_elev_deg']:.3f},{a['est_azim_deg']:.3f}) "
                    f"B=({b['est_dist']:.4f},{b['est_elev_deg']:.3f},{b['est_azim_deg']:.3f}) "
                    f"| gt_elev A={a['gt_elev_deg']:.1f} B={b['gt_elev_deg']:.1f}"
                )

    out_json = root / f"leak_test_{args.dataset}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    merged = []
    if out_json.exists():
        try:
            done_keys = {(r["method"], r["mode"]) for r in results}
            merged = [r for r in json.loads(out_json.read_text(encoding="utf-8")) if (r.get("method"), r.get("mode")) not in done_keys]
        except Exception:
            merged = []
    order = {m: i for i, m in enumerate(METHODS)}
    results = sorted(merged + results, key=lambda r: (order.get(r.get("method"), 99), r.get("mode") != "strict"))
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[report] {out_json}")
    print("[gate] " + ("PASS: no ground-truth dependence detected in strict mode." if all_ok else "FAIL: strict-mode estimates depend on GT labels."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
