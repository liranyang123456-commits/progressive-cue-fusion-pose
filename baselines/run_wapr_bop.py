#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAPR (ECCV 2026) baseline on the exported BOP frame packages, two modes:

  refine : WAPR as a *refiner* from the protocol's initial pose T0 of every rotation bin
           (wapr_w_mask x recipe.wapr_iters, then SAPR x recipe.sapr_iters) -- the matched
           wide-baseline refinement protocol shared with all refiners.
  full   : WAPR's own estimator (12 hypotheses from the mask, WAPR x3, SAPR x2, WBPS ranking),
           i.e. its global / multi-hypothesis mode; reported in a separate row.

Run inside the `wapr` conda env:
  python baselines/run_wapr_bop.py --root bop6dof_results/lmo --mode refine
  python baselines/run_wapr_bop.py --root bop6dof_results/lmo --mode full
Writes <root>/baseline_wapr_<mode>.jsonl with the same metric fields as results.jsonl.
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
sys.path.insert(0, str(ROOT))
from bop_metrics import compute_metrics, eval_points_trimesh, load_models_info, symmetry_transforms  # noqa: E402

DATASETS = {
    "lm": {"root": Path(os.environ.get("BOP_ROOT", "data/bop")) / "lm", "models": "models", "models_eval": "models_eval"},
    "lmo": {"root": Path(os.environ.get("BOP_ROOT", "data/bop")) / "lmo", "models": "models", "models_eval": "models_eval"},
    "tless": {"root": Path(os.environ.get("BOP_ROOT", "data/bop")) / "tless", "models": "models_cad", "models_eval": "models_eval"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", default="refine", choices=["refine", "full", "refine-from-ours"],
                    help="refine: from the exported T0 bins; full: WAPR's own 12-hypothesis estimator; "
                         "refine-from-ours: from the proposed method's estimate in results.jsonl (hypothesis generator + learned refiner)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--ours-file", default="results.jsonl", help="refine-from-ours: file with the starting poses (R_est, t_est per tag)")
    ap.add_argument("--out-name", default="", help="override the output file name")
    args = ap.parse_args()

    import cv2
    import torch
    import trimesh
    from wapr import WAPREstimator, recipe
    from wapr.estimator import center_from_mesh

    recipe.visualize = False
    root = Path(args.root)
    frames = sorted([d for d in root.iterdir() if d.is_dir() and (d / "protocol.json").exists()])
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    out_path = root / (args.out_name or ("baseline_wapr_from_ours.jsonl" if args.mode == "refine-from-ours" else f"baseline_wapr_{args.mode}.jsonl"))
    ours_init: dict = {}
    if args.mode == "refine-from-ours":
        rp = root / args.ours_file
        for line in rp.read_text(encoding="utf-8").splitlines():
            try:
                r0 = json.loads(line)
                ours_init[r0["tag"]] = r0
            except Exception:
                pass
        print(f"[wapr] {len(ours_init)} initial poses from the proposed method")
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                rec0 = json.loads(line)
                done.add((rec0["tag"], str(rec0.get("bin", "none"))))
            except Exception:
                pass

    est = WAPREstimator(device=args.device)
    mesh_cache: dict = {}
    pts_cache: dict = {}
    info_cache: dict = {}
    print(f"[wapr:{args.mode}] {len(frames)} frames; iters wapr={recipe.wapr_iters} sapr={recipe.sapr_iters}", flush=True)
    for fdir in frames:
        tag = fdir.name
        if args.mode == "full" and (tag, "none") in done:
            continue
        proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
        ds, obj = proto["dataset"], int(proto["obj_id"])
        cfg = DATASETS[ds]
        if obj not in mesh_cache:
            ply = cfg["root"] / cfg["models"] / f"obj_{obj:06d}.ply"
            m = trimesh.load(str(ply), force="mesh", process=False)
            m.apply_scale(0.001)  # mm -> m
            m.metadata["name"] = f"obj{obj}"
            mesh_cache[obj] = est.prepare_meshes([m])[0]
            pts_cache[obj] = eval_points_trimesh(cfg["root"] / cfg["models_eval"], cfg["root"] / cfg["models"], obj)
            info_cache[obj] = load_models_info(cfg["root"] / cfg["models"])[str(obj)]
        pose_mesh, pts, info = mesh_cache[obj], pts_cache[obj], info_cache[obj]
        syms = symmetry_transforms(info)
        diameter_m = float(info["diameter"]) / 1000.0
        obs = np.load(fdir / "observation.npz")
        K = obs["K"].astype(np.float32)
        rgb = obs["rgb"]  # uint8 RGB
        depth_m = (obs["depth"].astype(np.float32) / 1000.0)
        mask = (obs["mask"] > 0).astype(np.uint8) * 255
        R_gt, t_gt = np.array(proto["gt_R"]), np.array(proto["gt_t"])
        center = center_from_mesh(pose_mesh)
        t0_all = time.perf_counter()
        try:
            if args.mode == "full":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = est.estimate_one_category_one_instance(rgb, depth_m, K, pose_mesh, diameter_m, bbox_xywh=None, mask=mask)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                P = np.asarray(out["pose_4x4"], dtype=np.float64)
                R_est, t_est = P[:3, :3], P[:3, 3] * 1000.0
                met = compute_metrics(R_est, t_est, R_gt, t_gt, obs["K"], pts, info, syms)
                rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj,
                       "method": "wapr_full", "bin": "none", "init_rot_err_deg": None, "R_est": R_est.tolist(), "t_est": t_est.tolist(),
                       "score_6d": float(out.get("score_6d", 0.0)), "t_total_s": dt, **met}
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{tag}] full: e_R={met['e_R_deg']:.1f} e_t={met['e_t_mm']:.1f} ADD(-S)={met['add_s_metric']:.1f} {dt:.2f}s", flush=True)
            else:
                inits_run = proto["inits"]
                if args.mode == "refine-from-ours":
                    r0 = ours_init.get(tag)
                    if r0 is None:
                        print(f"[skip] {tag}: no estimate of the proposed method")
                        continue
                    inits_run = [{"bin": "ours", "R0": r0["R_est"], "t0": r0["t_est"], "rot_err_deg": float(r0["e_R_deg"])}]
                for init in inits_run:
                    if (tag, str(init["bin"])) in done:
                        continue
                    T0 = np.eye(4, dtype=np.float32)
                    T0[:3, :3] = np.array(init["R0"], dtype=np.float32)
                    T0[:3, 3] = np.array(init["t0"], dtype=np.float32) / 1000.0
                    pose_t = torch.as_tensor(T0[None], device=est.device, dtype=torch.float32)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    p1 = est._refine(est.nets["wapr_w_mask"], pose_t, rgb, depth_m, mask, K, pose_mesh, center, diameter_m,
                                     iters=recipe.wapr_iters, max_rot=recipe.wapr_max_rot_rad, crop_ratio=recipe.refine_crop_ratio, use_mask=True)
                    # SAPR takes the 6-channel crop pair (no mask channel), exactly as in WAPR's own recipe
                    p2 = est._refine(est.nets["sapr"], p1, rgb, depth_m, mask, K, pose_mesh, center, diameter_m,
                                     iters=recipe.sapr_iters, max_rot=recipe.sapr_max_rot_rad, crop_ratio=recipe.refine_crop_ratio, use_mask=False)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    P = p2[0].detach().float().cpu().numpy().astype(np.float64)
                    R_est, t_est = P[:3, :3], P[:3, 3] * 1000.0
                    met = compute_metrics(R_est, t_est, R_gt, t_gt, obs["K"], pts, info, syms)
                    rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj,
                           "method": "wapr_refine" if args.mode == "refine" else "ours_then_wapr", "bin": init["bin"], "init_rot_err_deg": init["rot_err_deg"],
                           "R_est": R_est.tolist(), "t_est": t_est.tolist(), "t_total_s": dt, **met}
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"[{tag}] bin {init['bin']:>7} init={init['rot_err_deg']:5.1f} -> e_R={met['e_R_deg']:5.1f} e_t={met['e_t_mm']:6.1f} ADD(-S)={met['add_s_metric']:6.1f} {dt:.2f}s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[error] {tag}: {e}", flush=True)
            if args.mode == "full":
                # the estimator raised (e.g. no usable depth inside the mask): count the frame as a failure
                # instead of silently dropping it from the averages
                rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj,
                       "method": "wapr_full", "bin": "none", "init_rot_err_deg": None, "failed": True, "error": str(e)[:200],
                       "R_est": None, "t_est": None, "t_total_s": time.perf_counter() - t0_all,
                       "e_R_deg": 180.0, "e_t_mm": float("inf"), "add": float("inf"), "adds": float("inf"), "add_s_metric": float("inf"),
                       "add_lt_0.1d": False, "mssd": float("inf"), "mspd": float("inf"), "ar_mssd": 0.0, "ar_mspd": 0.0,
                       "diameter": float(info["diameter"]), "symmetric": len(syms) > 1}
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
    print(f"[done] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
