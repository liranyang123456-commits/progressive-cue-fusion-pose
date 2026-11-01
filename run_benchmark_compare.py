#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-method camera pose benchmark: Ours v9 vs differentiable-rendering & soft-raster variants.

See BENCHMARK_BASELINES.md for method ↔ literature mapping.

Usage:
  python check_dr_backends.py
  python run_benchmark_compare.py --suite quick --dataset bunny --num-gt 3
  python run_benchmark_compare.py --suite dr --dataset bunny --num-gt 10
  python aggregate_benchmark_results.py --dataset bunny --num-gt 10
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("PIPELINE_PYTHON", "python")
MAIN = ROOT / "Exp_Camera_Optim_RT_Val_Beam_Search12291_Standford122.py"

V9_SHARED: dict[str, str] = {
    "VAL_RUN_PROFILE": "v9",
    "VAL_DETERMINISTIC": "1",
    "VAL_RANDOM_SEED": "12291",
    "VAL_PRECOMPUTE_GT": "1",
    "VAL_GLOBAL_MODE": "rand1024",
    "VAL_LOCAL_MODE": "beam",
    "VAL_RAND_TOPK": "16",
    "VAL_TOPK": "16",
    "VAL_RAND_SAMPLES_PER_RADIUS": "1024",
    "VAL_RAND_BATCH": "512",
    "VAL_LOCAL_BEAM_WIDTH": "48",
    "VAL_LOCAL_BEAM_LEVELS": "32",
    "VAL_LOCAL_BEAM_EARLY_STOP": "1",
    "VAL_SEARCH_ENABLE": "0",
    "VAL_LOCAL_BEAM_LAST_LAYER_END_SCORE": "0",
    "VAL_LOCAL_BEAM_FINAL_DR_ENABLE": "0",
    "VAL_SELECT_USE_VISUAL_SCORE": "1",
    "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
}

DATASETS: dict[str, dict] = {
    "bunny": {
        "model": os.path.join(os.environ.get("STANFORD_ROOT", "data/stanford"), "bun_zipper.ply"),
        "extra": {},
    },
    "bunny_res4": {
        "model": os.path.join(os.environ.get("STANFORD_ROOT", "data/stanford"), "bun_zipper_res4.ply"),
        "extra": {},
    },
    "dragon": {
        "model": os.path.join(os.environ.get("STANFORD_ROOT", "data/stanford"), "xyzrgb_dragon.ply"),
        "extra": {
            "VAL_MESH_SIMPLIFY": "1",
            "VAL_MESH_TARGET_TRIS": "80000",
            "VAL_RAND_BATCH": "256",
        },
    },
    # Additional Stanford meshes (pre-simplified to 80k triangles by tools/download_stanford_meshes.py).
    # Bounding radii: armadillo ~100 (mm), happy ~0.111 (m), lucy ~940 (mm) -> four orders of magnitude.
    "armadillo": {
        "model": str(ROOT / "benchmark_assets" / "armadillo_80k.ply"),
        "extra": {"VAL_RAND_BATCH": "256"},
    },
    "happy": {
        "model": str(ROOT / "benchmark_assets" / "happy_80k.ply"),
        "extra": {"VAL_RAND_BATCH": "256"},
    },
    "lucy": {
        "model": str(ROOT / "benchmark_assets" / "lucy_80k.ply"),
        "extra": {"VAL_RAND_BATCH": "256"},
    },
}

# Shared Adam-DR refine stage (same GT / mesh / loss family as main script).
_DR_ADAM = {
    "VAL_BASELINE_PIPELINE": "init_only",
    "VAL_SEARCH_ENABLE": "1",
    "VAL_OPTIM_METHOD": "adam",
    "VAL_REFINE_MAX_ITERS": "400",
    "VAL_REFINE_LR": "0.01",
    "VAL_LOSS_SCHEDULE": "1",
}

METHODS: dict[str, dict[str, str]] = {
    # --- Ours ---
    "ours_v9": {
        "VAL_SEARCH_ENABLE": "0",
        "VAL_BASELINE_PIPELINE": "",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
    },
    "ours_v10": {
        "VAL_RUN_PROFILE": "v10",
        "VAL_SEARCH_ENABLE": "0",
        "VAL_BASELINE_PIPELINE": "",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
    },
    # reduced budget (accuracy-time trade-off): M=20 shells x S=256 directions, K=8 seeds, beam B=24 x L=16 (~1/8 of v10)
    "ours_v10_fast": {
        "VAL_RUN_PROFILE": "v10",
        "VAL_SEARCH_ENABLE": "0",
        "VAL_BASELINE_PIPELINE": "",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
        "VAL_RAND_RADIUS_STEPS": "20",
        "VAL_RAND_SAMPLES_PER_RADIUS": "256",
        "VAL_RAND_TOPK": "8",
        "VAL_TOPK": "8",
        "VAL_LOCAL_BEAM_WIDTH": "24",
        "VAL_LOCAL_BEAM_LEVELS": "16",
    },
    # --- nvdiffrast DR baselines (same backend, different silhouette treatment) ---
    "adam_dr": {
        **_DR_ADAM,
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
        "VAL_SIL_AA": "1",
    },
    "adam_dr_hard": {
        **_DR_ADAM,
        "VAL_SIL_RENDER_MODE": "nvdiffrast_hard",
        "VAL_SIL_AA": "0",
    },
    "adam_dr_gauss_soft": {
        **_DR_ADAM,
        "VAL_SIL_RENDER_MODE": "gaussian_soft",
        "VAL_SIL_SOFT_BLUR_SIGMA": "1.5",
        "VAL_SIL_AA": "1",
    },
    "adam_dr_sil_only": {
        **_DR_ADAM,
        "VAL_REFINE_SIL_ONLY": "1",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
    },
    "hybrid_dr": {
        "VAL_BASELINE_PIPELINE": "init_only",
        "VAL_SEARCH_ENABLE": "1",
        "VAL_OPTIM_METHOD": "hybrid",
        "VAL_GREEDY_WARMUP_ITERS": "120",
        "VAL_REFINE_MAX_ITERS": "400",
        "VAL_REFINE_LR": "0.01",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
    },
    "hybrid_dr_gauss_soft": {
        "VAL_BASELINE_PIPELINE": "init_only",
        "VAL_SEARCH_ENABLE": "1",
        "VAL_OPTIM_METHOD": "hybrid",
        "VAL_GREEDY_WARMUP_ITERS": "120",
        "VAL_REFINE_MAX_ITERS": "400",
        "VAL_REFINE_LR": "0.01",
        "VAL_SIL_RENDER_MODE": "gaussian_soft",
        "VAL_SIL_SOFT_BLUR_SIGMA": "1.5",
    },
    "grid_adam": {
        "VAL_BASELINE_PIPELINE": "grid_only",
        "VAL_SEARCH_ENABLE": "1",
        "VAL_OPTIM_METHOD": "adam",
        "VAL_RAND_SAMPLES_PER_RADIUS": "0",
        "VAL_RAND_ADD_GRID_ANCHORS": "1",
        "VAL_RAND_TOPK": "8",
        "VAL_REFINE_MAX_ITERS": "400",
        "VAL_REFINE_LR": "0.01",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
    },
    "grid_adam_gauss_soft": {
        "VAL_BASELINE_PIPELINE": "grid_only",
        "VAL_SEARCH_ENABLE": "1",
        "VAL_OPTIM_METHOD": "adam",
        "VAL_RAND_SAMPLES_PER_RADIUS": "0",
        "VAL_RAND_ADD_GRID_ANCHORS": "1",
        "VAL_RAND_TOPK": "8",
        "VAL_REFINE_MAX_ITERS": "400",
        "VAL_REFINE_LR": "0.01",
        "VAL_SIL_RENDER_MODE": "gaussian_soft",
        "VAL_SIL_SOFT_BLUR_SIGMA": "1.5",
    },
    "rand_restart_adam": {
        "VAL_BASELINE_PIPELINE": "rand_adam",
        "VAL_SEARCH_ENABLE": "1",
        "VAL_OPTIM_METHOD": "adam",
        "VAL_MS_REFINE_ENABLE": "1",
        "VAL_MS_REFINE_TOPN": "8",
        "VAL_MS_REFINE_SHORT_ITERS": "150",
        "VAL_MS_REFINE_SHORT_METHOD": "adam",
        "VAL_REFINE_MAX_ITERS": "400",
        "VAL_REFINE_LR": "0.01",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
    },
    "grid_hybrid": {
        "VAL_BASELINE_PIPELINE": "grid_only",
        "VAL_SEARCH_ENABLE": "1",
        "VAL_OPTIM_METHOD": "hybrid",
        "VAL_GREEDY_WARMUP_ITERS": "80",
        "VAL_RAND_SAMPLES_PER_RADIUS": "0",
        "VAL_RAND_ADD_GRID_ANCHORS": "1",
        "VAL_RAND_TOPK": "8",
        "VAL_REFINE_MAX_ITERS": "400",
        "VAL_REFINE_LR": "0.01",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
    },
    "ours_v9_se3": {
        "VAL_SEARCH_ENABLE": "0",
        "VAL_SE3_REFINE_ENABLE": "1",
        "VAL_SE3_REFINE_ITERS": "120",
        "VAL_SE3_REFINE_LR": "0.003",
    },
    "ours_v9_geom": {
        "VAL_SEARCH_ENABLE": "0",
        "VAL_SELECT_USE_VISUAL_SCORE": "0",
    },
    "ours_v12c": {
        "VAL_RUN_PROFILE": "v12c",
        "VAL_SEARCH_ENABLE": "0",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_SCORE": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_COND": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_MIN_ELEV": "58.0",
        "VAL_LOCAL_BEAM_FINAL_DR_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_COND_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_MIN_SEED_ELEV": "58.0",
        "VAL_LOCAL_BEAM_FINAL_DR_STEPS": "4",
        "VAL_LOCAL_BEAM_FINAL_DR_LR": "0.012",
    },
    "ours_v12f": {
        "VAL_RUN_PROFILE": "v12f",
        "VAL_SEARCH_ENABLE": "0",
        "VAL_LOCAL_BEAM_COND_USE_GT_ELEV": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_SCORE": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_COND": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_MIN_ELEV": "72.0",
        "VAL_LOCAL_BEAM_LAST_LAYER_FD_TOPM": "4",
        "VAL_LOCAL_BEAM_FINAL_DR_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_COND_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_MIN_SEED_ELEV": "72.0",
        "VAL_LOCAL_BEAM_FINAL_DR_STEPS": "4",
        "VAL_LOCAL_BEAM_FINAL_DR_LR": "0.012",
    },
    "grid_adam_top16": {
        "VAL_BASELINE_PIPELINE": "grid_only",
        "VAL_SEARCH_ENABLE": "1",
        "VAL_OPTIM_METHOD": "adam",
        "VAL_RAND_SAMPLES_PER_RADIUS": "0",
        "VAL_RAND_ADD_GRID_ANCHORS": "1",
        "VAL_RAND_TOPK": "16",
        "VAL_REFINE_MAX_ITERS": "600",
        "VAL_REFINE_LR": "0.008",
        "VAL_MS_REFINE_ENABLE": "1",
        "VAL_MS_REFINE_TOPN": "8",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
    },
    # --- Ablations ---
    "greedy_bt_local": {
        "VAL_BASELINE_PIPELINE": "",
        "VAL_LOCAL_MODE": "greedy_bt",
        "VAL_SEARCH_ENABLE": "0",
    },
    "rand_topk_only": {
        "VAL_BASELINE_PIPELINE": "rand_only",
        "VAL_SEARCH_ENABLE": "0",
    },
    # w/o distance alignment under the v10 global search: no depth-guided pre-beam sweep,
    # no depth-hint distance penalty in seed ranking, no distance term in VisualSelect.
    "ours_v10_noda": {
        "VAL_RUN_PROFILE": "v10",
        "VAL_SEARCH_ENABLE": "0",
        "VAL_BASELINE_PIPELINE": "",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
        "VAL_LOCAL_BEAM_DIST_REFINE": "0",
        "VAL_REF_DIST_PENALTY_WEIGHT": "0.0",
        "VAL_SELECT_REF_DIST_WEIGHT": "0.0",
    },
    # Geometry-only final selection under v10 (RGB removed from VisualSelect).
    "ours_v10_geom": {
        "VAL_RUN_PROFILE": "v10",
        "VAL_SEARCH_ENABLE": "0",
        "VAL_BASELINE_PIPELINE": "",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
        "VAL_SELECT_USE_VISUAL_SCORE": "0",
    },
    # Equal-budget radius-schedule ablations (~41k proposals each, same seeds / beam):
    #   fixed multi-radius: legacy linear list 0.5r..4.0r (36 shells), not adapted to the depth hint
    #   single shell:       one shell at 2.0r with all 40960 directions
    "ours_v10_fixedmr": {
        "VAL_RUN_PROFILE": "v10",
        "VAL_SEARCH_ENABLE": "0",
        "VAL_BASELINE_PIPELINE": "",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
        "VAL_RAND_RADII_MULTS": ",".join(f"{0.5 + 0.1 * i:.1f}" for i in range(36)),
        "VAL_RAND_SAMPLES_PER_RADIUS": "1138",
    },
    "ours_v10_1shell": {
        "VAL_RUN_PROFILE": "v10",
        "VAL_SEARCH_ENABLE": "0",
        "VAL_BASELINE_PIPELINE": "",
        "VAL_SIL_RENDER_MODE": "nvdiffrast_aa",
        "VAL_RAND_RADII_MULTS": "2.0",
        "VAL_RAND_SAMPLES_PER_RADIUS": "40960",
    },
}

# Information Fusion 主文对比：Ours 最佳(v10) + v9 + 全部 DR/搜索基线
PAPER_METHODS: list[str] = [
    "ours_v10",
    "ours_v9",
    "grid_adam",
    "rand_restart_adam",
    "adam_dr",
    "adam_dr_hard",
    "adam_dr_gauss_soft",
    "adam_dr_sil_only",
    "hybrid_dr",
    "hybrid_dr_gauss_soft",
    "grid_adam_gauss_soft",
    "grid_hybrid",
    "ours_v9_se3",
]

SUITES: dict[str, list[str]] = {
    "quick": ["ours_v9", "ours_v10", "adam_dr", "grid_adam", "hybrid_dr"],
    "paper": PAPER_METHODS,
    "dr": [
        "ours_v9",
        "adam_dr",
        "adam_dr_hard",
        "adam_dr_gauss_soft",
        "adam_dr_sil_only",
        "hybrid_dr",
        "hybrid_dr_gauss_soft",
        "grid_adam",
        "grid_adam_gauss_soft",
        "rand_restart_adam",
        "grid_hybrid",
        "ours_v9_se3",
    ],
    "full": list(METHODS.keys()),
    "extended": [
        "greedy_bt_local",
        "rand_topk_only",
        "ours_v9_geom",
        "ours_v12c",
        "ours_v12f",
        "grid_adam_top16",
    ],
}

# $N{=}100$ full benchmark: best Ours first, then every in-framework method.
GT_CACHE_METHOD = "ours_v10"
FULL100_KAOLIN_METHODS: list[str] = [GT_CACHE_METHOD] + [
    m for m in METHODS.keys() if m != GT_CACHE_METHOD
]
SUITES["full100"] = FULL100_KAOLIN_METHODS

P3D_VARIANTS_100 = "p3d_softsil_spherical,p3d_softsil_xyz,p3d_softsil_grid,p3d_softsil_reg,p3d_softsil_gt_init"


def _gt_tag(i: int) -> str:
    return f"gt_{i:02d}"


def _gt_cache_complete(cache_dir: Path, num_gt: int) -> bool:
    if not (cache_dir / "gt_poses.csv").exists():
        return False
    for i in range(int(num_gt)):
        tag_dir = cache_dir / _gt_tag(i)
        if not (tag_dir / "gt_refs_search.npz").exists() or not (tag_dir / "gt_refs_full.npz").exists():
            return False
    return True


def _sync_gt_cache_from_reference(*, cache_dir: Path, out_dir: Path, num_gt: int) -> bool:
    """Copy pre-rendered GT folders from ours_v10 (or other cache) to skip GT precompute."""
    if not _gt_cache_complete(cache_dir, num_gt):
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("gt_poses.csv", "gt_poses_pose4x4.npy"):
        src = cache_dir / fname
        dst = out_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    for i in range(int(num_gt)):
        tag = _gt_tag(i)
        src = cache_dir / tag
        dst = out_dir / tag
        if not src.is_dir():
            return False
        if dst.exists():
            continue
        shutil.copytree(src, dst)
    return _gt_cache_complete(out_dir, num_gt)


def _finished_tags(out_dir: Path) -> set[str]:
    """Tags (gt_XX) that already have a row in pose_metrics_summary.csv."""
    csv_path = out_dir / "pose_metrics_summary.csv"
    if not csv_path.exists():
        return set()
    tags: set[str] = set()
    try:
        for ln in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
            t = ln.split(",")[0].strip()
            if t.startswith("gt_"):
                tags.add(t)
    except Exception:
        return set()
    return tags


def _run_is_complete(out_dir: Path, num_gt: int) -> bool:
    csv_path = out_dir / "pose_metrics_summary.csv"
    if not csv_path.exists():
        return False
    try:
        lines = [ln for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return len(lines) >= int(num_gt) + 1
    except Exception:
        return False


def run_method(
    *,
    dataset: str,
    method: str,
    num_gt: int,
    dry_run: bool,
    record: bool,
    skip_existing: bool = False,
    force: bool = False,
    max_views: int = 0,
) -> int:
    if dataset not in DATASETS:
        print(f"[Error] unknown dataset: {dataset}")
        return 2
    if method not in METHODS:
        print(f"[Error] unknown method: {method}")
        return 2

    ds = DATASETS[dataset]
    model = ds["model"]
    ds_extra = dict(ds.get("extra", {}))
    portable = ROOT / "benchmark_assets" / f"{dataset}_portable.ply"
    if not Path(model).exists() and portable.exists():
        # Worker machines: use the exact mesh exported by the reference run (already simplified).
        model = str(portable)
        ds_extra.pop("VAL_MESH_SIMPLIFY", None)
        ds_extra.pop("VAL_MESH_TARGET_TRIS", None)
        ds_extra["VAL_MESH_SIMPLIFY"] = "0"
        print(f"[Bench] using portable mesh {portable.name}")
    if not Path(model).exists():
        print(f"[Skip] model missing: {model}")
        return 0

    run_root = ROOT / "benchmark_runs" / f"{dataset}_{num_gt}gt"
    out_dir = run_root / method
    out_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and not force and _run_is_complete(out_dir, num_gt):
        print(f"[Skip] {dataset}/{method}: already has {num_gt} GT metrics")
        return 0
    log_path = out_dir / "run.log"

    cache_dir = run_root / GT_CACHE_METHOD
    if method != GT_CACHE_METHOD:
        if _sync_gt_cache_from_reference(cache_dir=cache_dir, out_dir=out_dir, num_gt=num_gt):
            print(f"[GT-cache] reused {GT_CACHE_METHOD} GT for {method} ({num_gt} views)")

    env = os.environ.copy()
    env.pop("VAL_GT_TAGS", None)
    env["BENCHMARK_PRESERVE_ENV"] = "1"
    env["FORCE_CODE_RUN_PARAMS"] = "1"
    env.update(V9_SHARED)
    env.update(ds_extra)
    env.update(METHODS[method])
    env["INPUT_MODEL_PATH"] = model
    env["OUTPUT_DIR"] = str(out_dir)
    env["NUM_GT_POSES"] = str(num_gt)
    if method != GT_CACHE_METHOD and _gt_cache_complete(out_dir, num_gt):
        env["VAL_PRECOMPUTE_GT"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    resume_tags: list[str] = []
    if not force:
        done = _finished_tags(out_dir)
        pending = [f"gt_{i:02d}" for i in range(int(num_gt)) if f"gt_{i:02d}" not in done]
        if done and not pending:
            print(f"[Skip] {dataset}/{method}: all {num_gt} GT views already finished")
            return 0
        if max_views and max_views > 0:
            # process only the first K views of the (seeded, deterministic) GT sequence now; later
            # invocations resume with the remaining ones so that N=100 runs can be built up in passes
            pending = pending[: int(max_views)]
            if not pending:
                print(f"[Skip] {dataset}/{method}: first {max_views} views already finished")
                return 0
        if done or (max_views and max_views > 0):
            resume_tags = pending
            env["VAL_GT_TAGS"] = ",".join(resume_tags)
            if _gt_cache_complete(out_dir, num_gt):
                env["VAL_PRECOMPUTE_GT"] = "0"
            print(f"[Bench] resume {dataset}/{method}: {len(done)} done, running {len(resume_tags)} view(s)")

    print(f"\n{'=' * 72}")
    print(f"[Bench] dataset={dataset} method={method} num_gt={num_gt}")
    print(f"        sil_mode={env.get('VAL_SIL_RENDER_MODE', '?')}")
    print(f"        out={out_dir}")
    print(f"{'=' * 72}")

    if dry_run:
        for k in sorted(env.keys()):
            if k.startswith("VAL_") or k in ("OUTPUT_DIR", "NUM_GT_POSES", "INPUT_MODEL_PATH"):
                print(f"  {k}={env[k]}")
        return 0

    meta = {
        "method": method,
        "dataset": dataset,
        "num_gt": num_gt,
        "started_at": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in env.items() if k.startswith("VAL_") or k in ("INPUT_MODEL_PATH",)},
    }
    try:
        (out_dir / "bench_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    with log_path.open("a" if (resume_tags and _finished_tags(out_dir)) else "w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            [PYTHON, str(MAIN)],
            cwd=str(ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    ok = proc.returncode == 0 and (out_dir / "pose_metrics_summary.csv").exists()
    print(f"[Bench] done method={method} ok={ok} exit={proc.returncode}")
    if ok and record:
        subprocess.run(
            [PYTHON, str(ROOT / "record_benchmark_results.py"), "--run-root", str(out_dir.parent)],
            cwd=str(ROOT),
            check=False,
        )
    return 0 if ok else int(proc.returncode or 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="DR / soft-raster pose benchmark runner")
    ap.add_argument("--dataset", default="bunny")
    ap.add_argument("--num-gt", type=int, default=10)
    ap.add_argument("--methods", default="", help="Comma-separated (overrides --suite)")
    ap.add_argument(
        "--suite",
        default="",
        help="quick | paper | dr | full — preset method lists",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if pose_metrics_summary has num_gt rows")
    ap.add_argument("--force", action="store_true", help="Re-run even when results exist")
    ap.add_argument("--record", action="store_true", help="Append results to benchmark_results/master.jsonl")
    ap.add_argument("--max-views", type=int, default=0, help="Run at most this many not-yet-finished views now (resumable passes)")
    ap.add_argument("--list-methods", action="store_true")
    args = ap.parse_args()

    if args.list_methods:
        print("Methods:")
        for mid, cfg in METHODS.items():
            sil = cfg.get("VAL_SIL_RENDER_MODE", "(search-only)")
            print(f"  {mid:24} sil={sil}")
        print("\nSuites:", ", ".join(SUITES.keys()))
        return 0

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    elif args.suite:
        methods = SUITES.get(args.suite, [])
        if not methods:
            print(f"[Error] unknown suite: {args.suite}")
            return 2
    else:
        methods = SUITES["dr"]

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    active_path = ROOT / "benchmark_results" / "active_suite.json"
    if not args.dry_run:
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(
            json.dumps(
                {
                    "suite": args.suite or "custom",
                    "dataset": args.dataset,
                    "num_gt": args.num_gt,
                    "methods": methods,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "status": "running",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    rc = 0
    for ds in datasets:
        for method in methods:
            rc |= run_method(
                dataset=ds,
                method=method,
                num_gt=args.num_gt,
                dry_run=args.dry_run,
                record=args.record,
                skip_existing=args.skip_existing,
                force=args.force,
                max_views=args.max_views,
            )
    if not args.dry_run:
        if active_path.parent.exists():
            try:
                active = json.loads(active_path.read_text(encoding="utf-8"))
                active["status"] = "completed" if rc == 0 else "failed"
                active["finished_at"] = datetime.now(timezone.utc).isoformat()
                active_path.write_text(json.dumps(active, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        if args.record:
            subprocess.run(
                [PYTHON, str(ROOT / "record_benchmark_results.py"), "--scan-all", "--report", "--export-csv"],
                cwd=str(ROOT),
                check=False,
            )
        if rc == 0:
            print("\n[Bench] Aggregate:")
            print(
                f"  python {ROOT / 'aggregate_benchmark_results.py'} "
                f"--dataset {args.dataset} --num-gt {args.num_gt}"
            )
            print(f"[Bench] Master log: {ROOT / 'benchmark_results' / 'master.jsonl'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
