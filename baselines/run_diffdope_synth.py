#!/usr/bin/env python3
"""
Diff-DOPE (Tremblay et al., 2023) on the synthetic Stanford benchmark: the same observations as every other
method of Tables 1-2 (tools/export_synth_frames.py), started from the benchmark's fixed initial camera, i.e.
the start of every DR baseline. Published hyper-parameters (60 iterations, batch 8, lr bounds [0.01, 100]),
mask + depth terms, mesh and pose in mm with object scale 0.01 and depth in cm -- exactly the configuration
used on BOP (baselines/run_diffdope_bop.py). --sanity starts from the ground truth (same schedule).

Run inside the `kaolin` env:  python baselines/run_diffdope_synth.py --mesh bunny
Writes benchmark_runs/<mesh>_<N>gt/diffdope/pose_metrics_summary.csv (+ records.jsonl).
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

F = np.diag([1.0, -1.0, -1.0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--radius-mm", type=float, default=100.0,
                    help="bounding radius the scene is scaled to; Diff-DOPE's learning rates are absolute (cm), so the scale is "
                         "chosen such that the optimisation is stable when started from the ground truth (see --sanity)")
    ap.add_argument("--tag", default="", help="suffix of the output method directory (calibration runs)")
    args = ap.parse_args()

    import torch
    import nvdiffrast.torch as dr
    from omegaconf import OmegaConf
    import diffdope as dd

    try:
        dr.RasterizeGLContext()
    except Exception:
        dr.RasterizeGLContext = dr.RasterizeCudaContext  # type: ignore[attr-defined]

    root = ROOT / "synth_frames" / args.mesh
    frames = sorted(d for d in root.iterdir() if d.is_dir() and (d / "protocol.json").exists())
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    out = SummaryWriter(args.mesh, ("diffdope_sanity" if args.sanity else "diffdope") + args.tag)
    k = args.radius_mm / 100.0  # the exported frames use a 100 mm radius
    proto0 = json.loads((frames[0] / "protocol.json").read_text(encoding="utf-8"))
    mesh_path = proto0["mesh_path"]
    if abs(k - 1.0) > 1e-9:
        import trimesh
        mesh_path = str(root / f"model_r{args.radius_mm:g}mm.ply")
        if not Path(mesh_path).exists():
            mm = trimesh.load(proto0["mesh_path"], force="mesh", process=False)
            mm.vertices = mm.vertices * k
            mm.export(mesh_path)
    iters = args.iters  # the sanity run uses the full schedule from the ground truth, as on BOP
    for fdir in frames:
        tag = fdir.name
        if tag in out.done:
            continue
        proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
        obs = np.load(fdir / "observation.npz")
        K = obs["K"].astype(np.float64)
        H, W = obs["mask"].shape
        rgb = np.flipud(obs["rgb"].astype(np.float32) / 255.0).copy()
        depth_dd = np.flipud(obs["depth"].astype(np.float32) * k / 100.0).copy()  # mm -> cm (object scale 0.01)
        seg = np.flipud(np.repeat((obs["mask"] > 0).astype(np.float32)[..., None], 3, axis=-1)).copy()
        R0 = proto["gt_R"] if args.sanity else proto["inits"][0]["R0"]
        t0v = (np.array(proto["gt_t"] if args.sanity else proto["inits"][0]["t0"]) * k).tolist()
        cfg = OmegaConf.create({
            "camera": {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]), "im_width": int(W), "im_height": int(H), "znear": 0.1, "zfar": 10000.0},
            "scene": {"path_img": None, "path_depth": None, "path_segmentation": None, "image_resize": 1.0},
            "object3d": {"position": [float(x) for x in t0v], "rotation": [float(x) for x in np.array(R0).reshape(-1)], "model_path": mesh_path, "scale": 0.01, "batchsize": args.batch},
            "losses": {"l1_rgb_with_mask": False, "weight_rgb": 0.7, "l1_depth_with_mask": True, "weight_depth": 1.0, "l1_mask": True, "weight_mask": 1.0},
            "hyperparameters": {"nb_iterations": int(iters), "batchsize": args.batch, "base_lr": 20, "learning_rates_bound": [0.01, 100], "learning_rate_base": 1, "lr_decay": 0.1},
            "render_images": {"nrow": 4, "final_width_batch": 2000, "add_background": True, "alpha_overlay": 0.7, "add_countour": True, "color_countour": [0.46, 0.73, 0], "flip_result": True, "crop_around_mask": True},
        })
        scene = dd.Scene(image_resize=1.0)
        scene.tensor_rgb = dd.Image(img_tensor=torch.tensor(rgb).float(), flip_img=False)
        scene.tensor_depth = dd.Image(img_tensor=torch.tensor(depth_dd).float(), flip_img=False, depth=True)
        scene.tensor_segmentation = dd.Image(img_tensor=torch.tensor(seg).float(), flip_img=False)
        scene.path_img = scene.path_depth = scene.path_segmentation = "tensor"
        t_start = time.perf_counter()
        ddope = None
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ddope = dd.DiffDope(cfg=cfg, scene=scene)
            ddope.run_optimization()
            P_gl = np.asarray(ddope.get_pose(), dtype=np.float64)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            R_est, t_est = F @ P_gl[:3, :3], (F @ P_gl[:3, 3]) / 0.01 / k  # internal cm -> mm of the exported 100 mm frame
            met = synth_metrics(R_est, t_est, proto)
            out.write(tag, met, dt, {"R_est": R_est.tolist(), "t_est_mm": t_est.tolist()})
            print(f"[{args.mesh} {tag}] e_R={met['rot_err_deg']:6.1f}  |dd|/d={met['dist_err'] / max(1e-12, met['gt_dist']):.3f}  {dt:.1f}s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            out.write(tag, failed_metrics(proto), time.perf_counter() - t_start, {"failed": True, "error": str(e)[:200]})
            print(f"[{args.mesh} {tag}] failed: {e}", flush=True)
        finally:
            del ddope
            torch.cuda.empty_cache()
    print(f"[done] {out.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())