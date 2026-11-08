#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diff-DOPE (Tremblay et al., 2023; training-free differentiable-rendering refiner) on the
exported BOP frame packages, started from the protocol's initial pose T0 of every rotation bin.

Diff-DOPE optimises a batch of B pose copies with random learning rates (SGD on quaternion +
translation) against mask / depth / RGB L1 terms and returns the lowest-loss member.
We use its published hyper-parameters (60 iterations, batch 8, lr bounds [0.01, 100]) with the
mask + depth terms (RGB off: the T-LESS CAD models carry no texture and our method also uses
geometry only on real data). Units: mesh mm -> cm (scale 0.01), depth cm, as in its BOP example.

Run inside the `kaolin` env (nvdiffrast available):
  python baselines/run_diffdope_bop.py --root bop6dof_results/lmo
  python baselines/run_diffdope_bop.py --root bop6dof_results/lmo --sanity   # start from GT, 1 iteration
  python baselines/run_diffdope_bop.py --root bop6dof_results/lmo --from-ours  # from the proposed method's estimate
Writes <root>/baseline_diffdope.jsonl (same metric fields as results.jsonl).
"""
from __future__ import annotations

import os

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_metrics import compute_metrics, eval_points, load_models_info, symmetry_transforms  # noqa: E402

DATASETS = {
    "lm": {"root": Path(os.environ.get("BOP_ROOT", "data/bop")) / "lm", "models": "models", "models_eval": "models_eval"},
    "lmo": {"root": Path(os.environ.get("BOP_ROOT", "data/bop")) / "lmo", "models": "models", "models_eval": "models_eval"},
    "tless": {"root": Path(os.environ.get("BOP_ROOT", "data/bop")) / "tless", "models": "models_cad", "models_eval": "models_eval"},
}
F = np.diag([1.0, -1.0, -1.0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--from-ours", action="store_true",
                    help="start from the proposed method's estimate in results.jsonl (no T0) instead of the T0 bins")
    ap.add_argument("--rgb", action="store_true",
                    help="also enable Diff-DOPE's masked RGB L1 term (vertex colours of the BOP model) from the first iteration")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    import torch
    import nvdiffrast.torch as dr
    from omegaconf import OmegaConf
    import diffdope as dd

    # Windows builds of nvdiffrast used here provide the CUDA rasteriser; Diff-DOPE asks for the GL one.
    try:
        dr.RasterizeGLContext()
    except Exception:
        dr.RasterizeGLContext = dr.RasterizeCudaContext  # type: ignore[attr-defined]

    root = Path(args.root)
    frames = sorted([d for d in root.iterdir() if d.is_dir() and (d / "protocol.json").exists()])
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if args.from_ours:
        out_path = root / ("baseline_diffdope_rgb_from_ours.jsonl" if args.rgb else "baseline_diffdope_from_ours.jsonl")
        ours_init: dict = {}
        for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                r0 = json.loads(line)
                ours_init[r0["tag"]] = r0
            except Exception:
                pass
    else:
        out_path = root / ("baseline_diffdope_sanity.jsonl" if args.sanity else ("baseline_diffdope_rgb.jsonl" if args.rgb else "baseline_diffdope.jsonl"))
    done = set()  # (tag, bin) pairs already evaluated -> resumable at bin granularity
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                rec0 = json.loads(line)
                done.add((rec0["tag"], str(rec0.get("bin"))))
            except Exception:
                pass

    pts_cache: dict = {}
    info_cache: dict = {}
    for fdir in frames:
        tag = fdir.name
        proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
        ds, obj = proto["dataset"], int(proto["obj_id"])
        cfg_ds = DATASETS[ds]
        ply = cfg_ds["root"] / cfg_ds["models"] / f"obj_{obj:06d}.ply"
        if obj not in pts_cache:
            pts_cache[obj] = eval_points(cfg_ds["root"] / cfg_ds["models_eval"], cfg_ds["root"] / cfg_ds["models"], obj)
            info_cache[obj] = load_models_info(cfg_ds["root"] / cfg_ds["models"])[str(obj)]
        pts, info = pts_cache[obj], info_cache[obj]
        syms = symmetry_transforms(info)
        obs = np.load(fdir / "observation.npz")
        K = obs["K"].astype(np.float64)
        H, W = obs["mask"].shape
        rgb = np.flipud(obs["rgb"].astype(np.float32) / 255.0).copy()
        # Diff-DOPE's internal unit is (input unit) x object3d.scale: BOP meshes and poses are in mm and we
        # use scale=0.01 (as the authors' example, whose mesh/pose are in mm and whose depth PNG is divided by
        # 100), so the observed depth must be given in the same unit, i.e. mm / 100.
        depth_dd = np.flipud(obs["depth"].astype(np.float32) / 100.0).copy()
        seg = np.flipud(np.repeat((obs["mask"] > 0).astype(np.float32)[..., None], 3, axis=-1)).copy()
        R_gt, t_gt = np.array(proto["gt_R"]), np.array(proto["gt_t"])
        if args.from_ours:
            r0 = ours_init.get(tag)
            if r0 is None:
                print(f"[skip] {tag}: no estimate of the proposed method")
                continue
            inits = [{"bin": "ours", "R0": r0["R_est"], "t0": r0["t_est"], "rot_err_deg": float(r0["e_R_deg"])}]
        elif args.sanity:
            # ground-truth start plus the mildest exported perturbation: both must end close to the ground truth
            inits = [{"bin": "gt", "R0": R_gt.tolist(), "t0": t_gt.tolist(), "rot_err_deg": 0.0}] + [i for i in proto["inits"] if i["bin"] == "15-30"]
        else:
            inits = proto["inits"]
        for init in inits:
            if (tag, str(init["bin"])) in done:
                continue
            cfg = OmegaConf.create({
                "camera": {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]), "im_width": int(W), "im_height": int(H), "znear": 0.1, "zfar": 1000.0},
                "scene": {"path_img": None, "path_depth": None, "path_segmentation": None, "image_resize": 1.0},
                "object3d": {"position": [float(x) for x in init["t0"]], "rotation": [float(x) for x in np.array(init["R0"]).reshape(-1)], "model_path": str(ply), "scale": 0.01, "batchsize": args.batch},
                "losses": {"l1_rgb_with_mask": bool(args.rgb), "weight_rgb": 0.7, "l1_depth_with_mask": True, "weight_depth": 1.0, "l1_mask": True, "weight_mask": 1.0},
                "hyperparameters": {"nb_iterations": int(args.iters), "batchsize": args.batch, "base_lr": 20, "learning_rates_bound": [0.01, 100], "learning_rate_base": 1, "lr_decay": 0.1},
                "render_images": {"nrow": 4, "final_width_batch": 2000, "add_background": True, "alpha_overlay": 0.7, "add_countour": True, "color_countour": [0.46, 0.73, 0], "flip_result": True, "crop_around_mask": True},
            })
            scene = dd.Scene(image_resize=1.0)
            scene.tensor_rgb = dd.Image(img_tensor=torch.tensor(rgb).float(), flip_img=False)
            scene.tensor_depth = dd.Image(img_tensor=torch.tensor(depth_dd).float(), flip_img=False, depth=True)
            scene.tensor_segmentation = dd.Image(img_tensor=torch.tensor(seg).float(), flip_img=False)
            scene.path_img = scene.path_depth = scene.path_segmentation = "tensor"  # mark as present
            # the first call of a process may fail while Diff-DOPE JIT-compiles its transform plugin (no ninja on PATH);
            # it then falls back to its PyTorch path, so the same run is attempted once more
            for attempt in (1, 2):
                try:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    ddope = dd.DiffDope(cfg=cfg, scene=scene)
                    ddope.run_optimization()
                    P_gl = np.asarray(ddope.get_pose(), dtype=np.float64)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    R_est = F @ P_gl[:3, :3]
                    t_est = (F @ P_gl[:3, 3]) / 0.01  # cm -> mm
                    met = compute_metrics(R_est, t_est, R_gt, t_gt, K, pts, info, syms)
                    rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj,
                           "method": ("ours_then_" if args.from_ours else "") + ("diffdope_rgb" if args.rgb else "diffdope"), "bin": init["bin"], "init_rot_err_deg": init["rot_err_deg"],
                           "R_est": R_est.tolist(), "t_est": t_est.tolist(), "t_total_s": dt, "iters": int(cfg.hyperparameters.nb_iterations), **met}
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"[{tag}] bin {init['bin']:>7} init={init['rot_err_deg']:5.1f} -> e_R={met['e_R_deg']:5.1f} e_t={met['e_t_mm']:6.1f} ADD(-S)={met['add_s_metric']:6.1f} {dt:.1f}s", flush=True)
                    break
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[error] {tag} {init['bin']} (attempt {attempt}): {e}", flush=True)
                finally:
                    try:
                        del ddope
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
    print(f"[done] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
