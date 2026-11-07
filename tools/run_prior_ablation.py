#!/usr/bin/env python3
"""
Hemisphere-agnostic ablation of the seed priors on the views where WAPR's twelve-hypothesis estimator fails.

The ground-truth cameras of the synthetic benchmark lie on the upper hemisphere; the seed score contains a one-sided
elevation penalty (psi_elev) and a mesh-top-normal prior (psi_normal), and the diversity bins and grid anchors cover
elevations 0-90 deg only. `ours_v10_fast_sym` removes both priors and makes bins and anchors symmetric about the
equator (same number of bins and anchors), at the 1/8 budget of `ours_v10_fast`; `ours_v10_fast_ref` (the settings of
`ours_v10_fast` in a separate directory) is run on the same views as the budget-matched reference.

  python tools/run_prior_ablation.py --method ours_v10_fast_sym
  python tools/run_prior_ablation.py --method ours_v10_fast_ref
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_benchmark_compare as rbc  # noqa: E402

rbc.METHODS["ours_v10_fast_ref"] = dict(rbc.METHODS["ours_v10_fast"])  # separate directory, same settings
rbc.METHODS["ours_v10_fast_sym"] = {
    **rbc.METHODS["ours_v10_fast"],
    "VAL_RAND_ELEV_LAMBDA": "0",
    "VAL_NORMAL_PRIOR_WEIGHT": "0",
    "VAL_RAND_ELEV_BINS": "-90,-60,-30,0,30,60,90",
    "VAL_RAND_ANCHOR_ELEVS": "-75,-50,-25,0,25,50,75",
}

# views on which WAPR's estimator misses 10 deg (20 of 350)
VIEWS = {
    "bunny": (100, ["gt_05", "gt_07", "gt_16", "gt_39", "gt_72", "gt_76"]),
    "dragon": (100, ["gt_13", "gt_35", "gt_39", "gt_49", "gt_81", "gt_92"]),
    "armadillo": (50, ["gt_05", "gt_18", "gt_25", "gt_49"]),
    "happy": (50, ["gt_05", "gt_07", "gt_36", "gt_44"]),
}


def copy_gt(src_root: Path, dst_root: Path, tags: list[str]) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for fn in ("gt_poses.csv", "gt_poses_pose4x4.npy"):
        if (src_root / fn).exists() and not (dst_root / fn).exists():
            shutil.copy2(src_root / fn, dst_root / fn)
    for t in tags:
        (dst_root / t).mkdir(exist_ok=True)
        for f in (src_root / t).glob("gt_*"):
            if f.is_file() and not (dst_root / t / f.name).exists():
                shutil.copy2(f, dst_root / t / f.name)


def run(dataset: str, num_gt: int, method: str, tags: list[str]) -> int:
    ds = rbc.DATASETS[dataset]
    run_root = ROOT / "benchmark_runs" / f"{dataset}_{num_gt}gt"
    out_dir = run_root / method
    copy_gt(run_root / rbc.GT_CACHE_METHOD, out_dir, tags)
    pending = [t for t in tags if t not in rbc._finished_tags(out_dir)]
    if not pending:
        print(f"[skip] {dataset}/{method}: all requested views done")
        return 0
    env = os.environ.copy()
    env["BENCHMARK_PRESERVE_ENV"] = "1"
    env["FORCE_CODE_RUN_PARAMS"] = "1"
    env.update(rbc.V9_SHARED)
    env.update(dict(ds.get("extra", {})))
    env.update(rbc.METHODS[method])
    env.update({"INPUT_MODEL_PATH": ds["model"], "OUTPUT_DIR": str(out_dir), "NUM_GT_POSES": str(num_gt),
                "VAL_PRECOMPUTE_GT": "0", "VAL_GT_TAGS": ",".join(pending), "VAL_STRICT_NO_GT": "1",
                "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    print(f"[run] {dataset}/{method}: {len(pending)} view(s) {pending}", flush=True)
    with (out_dir / "run.log").open("a", encoding="utf-8") as f:
        proc = subprocess.run([rbc.PYTHON, str(rbc.MAIN)], cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"[done] {dataset}/{method} exit={proc.returncode}", flush=True)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ours_v10_fast_sym", "ours_v10_fast_ref"])
    args = ap.parse_args()
    for dataset, (num_gt, tags) in VIEWS.items():
        run(dataset, num_gt, args.method, tags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
