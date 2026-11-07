#!/usr/bin/env python3
"""
FoundationPose (CVPR 2024, model-based setting) on the synthetic Stanford benchmark: the same observations as every
other method of Tables 1-2 (640x480 RGB, metric depth and silhouette of the rendered target view;
tools/export_synth_frames.py).

  full   : its own registration (icosphere hypotheses refined for 5 iterations, ranked by its learned scorer), no initial pose
  refine : its refiner (5 iterations) from the benchmark's fixed initial camera, the start of every DR baseline; the rotation
           of that camera is kept and the translation re-initialised from the depth inside the mask, as its registration
           does for every hypothesis (same rule as on the BOP frames, baselines/run_foundationpose_bop.py)
  --sanity runs the refiner from the ground truth (must stay close to it)

Run in the `py3d` env (native Windows port, see run_foundationpose_bop.py):
  python baselines/run_foundationpose_synth.py --mesh bunny --mode full
Writes benchmark_runs/<mesh>_<N>gt/foundationpose_<mode>/pose_metrics_summary.csv (+ records.jsonl).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FP_DIR = Path(os.environ.get("FOUNDATIONPOSE_DIR", "external/FoundationPose"))
sys.path.insert(0, str(ROOT / "baselines"))
from synth_common import SummaryWriter, failed_metrics, synth_metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "refine"])
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(FP_DIR))
    sys.path.insert(0, str(FP_DIR / "mycpp" / "build"))
    import logging

    import nvdiffrast.torch as dr
    import torch
    import trimesh
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from Utils import set_seed

    logging.getLogger().setLevel(logging.WARNING)
    root = ROOT / "synth_frames" / args.mesh
    frames = sorted(d for d in root.iterdir() if d.is_dir() and (d / "protocol.json").exists())
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    method = "foundationpose_sanity" if args.sanity else f"foundationpose_{args.mode}"
    out = SummaryWriter(args.mesh, method)
    proto0 = json.loads((frames[0] / "protocol.json").read_text(encoding="utf-8"))
    mesh = trimesh.load(proto0["mesh_path"], force="mesh", process=False)
    mesh.apply_scale(0.001)  # mm -> m
    est = FoundationPose(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh,
                         scorer=ScorePredictor(), refiner=PoseRefinePredictor(), glctx=dr.RasterizeCudaContext(), debug=0,
                         debug_dir=str(ROOT / "bop6dof_results" / "_fp_debug"))
    tf_c = est.get_tf_to_centered_mesh().data.cpu().numpy()
    print(f"[foundationpose:{method}] {args.mesh}: {len(frames)} views", flush=True)
    for fdir in frames:
        tag = fdir.name
        if tag in out.done:
            continue
        proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
        obs = np.load(fdir / "observation.npz")
        K = np.asarray(obs["K"], dtype=np.float64)
        rgb = np.ascontiguousarray(obs["rgb"])
        depth_m = obs["depth"].astype(np.float32) / 1000.0
        mask = obs["mask"] > 0
        t_start = time.perf_counter()
        set_seed(0)
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if args.mode == "full" and not args.sanity:
                P = est.register(K=K, rgb=rgb, depth=depth_m, ob_mask=mask, iteration=args.iters)
            else:
                T0 = np.eye(4)
                T0[:3, :3] = np.array(proto["gt_R"] if args.sanity else proto["inits"][0]["R0"], dtype=np.float64)
                T0[:3, 3] = np.array(proto["gt_t"] if args.sanity else proto["inits"][0]["t0"], dtype=np.float64) / 1000.0
                Tc = T0 @ np.linalg.inv(tf_c)
                if not args.sanity:
                    Tc[:3, 3] = est.guess_translation(depth=depth_m, mask=mask, K=K)
                est.pose_last = torch.as_tensor(Tc, device="cuda", dtype=torch.float)
                P = est.track_one(rgb=rgb, depth=depth_m, K=K, iteration=args.iters)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            P = np.asarray(P, dtype=np.float64)
            met = synth_metrics(P[:3, :3], P[:3, 3] * 1000.0, proto)
            out.write(tag, met, dt, {"R_est": P[:3, :3].tolist(), "t_est_mm": (P[:3, 3] * 1000.0).tolist()})
            print(f"[{args.mesh} {tag}] e_R={met['rot_err_deg']:6.1f}  |dd|/d={met['dist_err'] / max(1e-12, met['gt_dist']):.3f}  {dt:.2f}s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            out.write(tag, failed_metrics(proto), time.perf_counter() - t_start, {"failed": True, "error": str(e)[:200]})
            print(f"[{args.mesh} {tag}] failed: {e}", flush=True)
    print(f"[done] {out.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
