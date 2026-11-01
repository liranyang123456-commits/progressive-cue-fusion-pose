#!/usr/bin/env python3
"""
WAPR (ECCV 2026) on the synthetic Stanford benchmark: the same observations as every other method of Tables 1-2
(640x480 RGB, metric depth and silhouette of the rendered target view; tools/export_synth_frames.py).

  full   : WAPR's own estimator (12 hypotheses from the mask, WAPR x3, SAPR x2, WBPS ranking), no initial pose
  refine : WAPR refiner (wapr_w_mask x recipe.wapr_iters, then SAPR x recipe.sapr_iters) from the benchmark's
           fixed initial camera, i.e. the start of every DR baseline
  --sanity runs `refine` from the ground truth (must stay close to it)

Run inside the `wapr` conda env:  python baselines/run_wapr_synth.py --mesh bunny --mode full
Writes benchmark_runs/<mesh>_<N>gt/wapr_<mode>/pose_metrics_summary.csv (+ records.jsonl).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines"))
from synth_common import SummaryWriter, failed_metrics, synth_metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "refine"])
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    import torch
    import trimesh
    from wapr import WAPREstimator, recipe
    from wapr.estimator import center_from_mesh

    recipe.visualize = False
    root = ROOT / "synth_frames" / args.mesh
    frames = sorted(d for d in root.iterdir() if d.is_dir() and (d / "protocol.json").exists())
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    method = "wapr_sanity" if args.sanity else f"wapr_{args.mode}"
    out = SummaryWriter(args.mesh, method)
    est = WAPREstimator(device=args.device)
    proto0 = json.loads((frames[0] / "protocol.json").read_text(encoding="utf-8"))
    m = trimesh.load(proto0["mesh_path"], force="mesh", process=False)
    m.apply_scale(0.001)  # mm -> m
    m.metadata["name"] = args.mesh
    pose_mesh = est.prepare_meshes([m])[0]
    center = center_from_mesh(pose_mesh)
    diameter_m = float(proto0["diameter_mm"]) / 1000.0
    print(f"[wapr:{method}] {args.mesh}: {len(frames)} views, diameter {diameter_m:.3f} m", flush=True)
    for fdir in frames:
        tag = fdir.name
        if tag in out.done:
            continue
        proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
        obs = np.load(fdir / "observation.npz")
        K = obs["K"].astype(np.float32)
        rgb = obs["rgb"]
        depth_m = obs["depth"].astype(np.float32) / 1000.0
        mask = (obs["mask"] > 0).astype(np.uint8) * 255
        t_start = time.perf_counter()
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if args.mode == "full" and not args.sanity:
                res = est.estimate_one_category_one_instance(rgb, depth_m, K, pose_mesh, diameter_m, bbox_xywh=None, mask=mask)
                P = np.asarray(res["pose_4x4"], dtype=np.float64)
            else:
                R0 = np.array(proto["gt_R"] if args.sanity else proto["inits"][0]["R0"], dtype=np.float32)
                t0v = np.array(proto["gt_t"] if args.sanity else proto["inits"][0]["t0"], dtype=np.float32) / 1000.0
                T0 = np.eye(4, dtype=np.float32)
                T0[:3, :3], T0[:3, 3] = R0, t0v
                pose_t = torch.as_tensor(T0[None], device=est.device, dtype=torch.float32)
                p1 = est._refine(est.nets["wapr_w_mask"], pose_t, rgb, depth_m, mask, K, pose_mesh, center, diameter_m,
                                 iters=recipe.wapr_iters, max_rot=recipe.wapr_max_rot_rad, crop_ratio=recipe.refine_crop_ratio, use_mask=True)
                p2 = est._refine(est.nets["sapr"], p1, rgb, depth_m, mask, K, pose_mesh, center, diameter_m,
                                 iters=recipe.sapr_iters, max_rot=recipe.sapr_max_rot_rad, crop_ratio=recipe.refine_crop_ratio, use_mask=False)
                P = p2[0].detach().float().cpu().numpy().astype(np.float64)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
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