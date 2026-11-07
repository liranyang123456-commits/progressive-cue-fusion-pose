#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoundationPose (CVPR 2024, model-based setting) on the exported BOP frame packages, three modes:

  full             : its own estimator (icosphere viewpoints x in-plane rotations clustered under the object's
                     symmetries, refined for 5 iterations, ranked by its learned scorer) -- initialisation-free,
                     reported next to WAPR's twelve-hypothesis mode.
  refine           : its refiner (5 iterations, no scorer) from the protocol's T0 of every rotation bin.
  refine-from-ours : its refiner from the proposed method's estimate in results.jsonl.

Released weights (refiner 2023-10-28-18-33-37, scorer 2024-01-11-20-02-45) under $FOUNDATIONPOSE_DIR/weights.
Native Windows port: the C++ extension mycpp.cluster_poses is replaced by the equivalent NumPy function in
$FOUNDATIONPOSE_DIR/mycpp/build/mycpp.py; everything else is the released code. Run in the `fpose` env:
  python baselines/run_foundationpose_bop.py --root bop6dof_results/lmo --mode full
Writes <root>/baseline_foundationpose_<mode>.jsonl with the same metric fields as results.jsonl.
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
sys.path.insert(0, str(ROOT))
from bop_metrics import compute_metrics, eval_points_trimesh, load_models_info, symmetry_transforms  # noqa: E402

BOP_ROOT = Path(os.environ.get("BOP_ROOT", "data/bop"))
DATASETS = {
    "lm": {"root": BOP_ROOT / "lm", "models": "models", "models_eval": "models_eval"},
    "lmo": {"root": BOP_ROOT / "lmo", "models": "models", "models_eval": "models_eval"},
    "tless": {"root": BOP_ROOT / "tless", "models": "models_cad", "models_eval": "models_eval"},
}
OUT_NAME = {"full": "baseline_foundationpose_full.jsonl", "refine": "baseline_foundationpose_refine.jsonl",
            "refine-from-ours": "baseline_foundationpose_from_ours.jsonl"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", default="full", choices=list(OUT_NAME))
    ap.add_argument("--iters", type=int, default=5, help="refiner iterations (5 = the released register() setting)")
    ap.add_argument("--t0-translation", default="fp", choices=["fp", "protocol"],
                    help="refine mode: 'fp' keeps the rotation of T0 and sets the translation by FoundationPose's own depth-centre "
                         "estimate, as its register() does for every hypothesis (SDFR likewise re-centres on the observed points); "
                         "'protocol' uses the perturbed translation of T0 unchanged")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--bins", default="", help="refine mode: comma-separated subset of the protocol's rotation bins (default: all)")
    args = ap.parse_args()

    sys.path.insert(0, str(FP_DIR))
    sys.path.insert(0, str(FP_DIR / "mycpp" / "build"))
    import logging

    import nvdiffrast.torch as dr
    import torch
    import trimesh
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from Utils import set_seed, symmetry_tfs_from_info

    logging.getLogger().setLevel(logging.WARNING)
    root = Path(args.root)
    frames = sorted(d for d in root.iterdir() if d.is_dir() and (d / "protocol.json").exists())
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    out_path = root / OUT_NAME[args.mode]
    if args.mode == "refine" and args.t0_translation == "protocol":
        out_path = root / "baseline_foundationpose_refine_t0trans.jsonl"
    if args.iters != 5:
        out_path = out_path.with_name(out_path.stem + f"_it{args.iters}.jsonl")
    ours_init = {}
    if args.mode == "refine-from-ours":
        for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                r0 = json.loads(line)
                ours_init[r0["tag"]] = r0
            except Exception:
                pass
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                rec0 = json.loads(line)
                done.add((rec0["tag"], str(rec0.get("bin", "none"))))
            except Exception:
                pass

    glctx = dr.RasterizeCudaContext()
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    est, cur_obj = None, None
    pts_cache, info_cache = {}, {}
    print(f"[foundationpose:{args.mode}] {len(frames)} frames, refiner iterations={args.iters}", flush=True)
    for fdir in frames:
        tag = fdir.name
        proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
        ds, obj = proto["dataset"], int(proto["obj_id"])
        if args.mode == "full":
            inits = [{"bin": "none", "rot_err_deg": None}]
        elif args.mode == "refine":
            inits = [i for i in proto["inits"] if not args.bins or i["bin"] in args.bins.split(",")]
        else:
            r0 = ours_init.get(tag)
            if r0 is None:
                print(f"[skip] {tag}: no estimate of the proposed method")
                continue
            inits = [{"bin": "ours", "R0": r0["R_est"], "t0": r0["t_est"], "rot_err_deg": float(r0["e_R_deg"])}]
        inits = [i for i in inits if (tag, str(i["bin"])) not in done]
        if not inits:
            continue
        cfg = DATASETS[ds]
        if (ds, obj) not in info_cache:
            pts_cache[(ds, obj)] = eval_points_trimesh(cfg["root"] / cfg["models_eval"], cfg["root"] / cfg["models"], obj)
            info_cache[(ds, obj)] = load_models_info(cfg["root"] / cfg["models"])[str(obj)]
        pts, info = pts_cache[(ds, obj)], info_cache[(ds, obj)]
        syms = symmetry_transforms(info)
        if cur_obj != (ds, obj):
            mesh = trimesh.load(str(cfg["root"] / cfg["models"] / f"obj_{obj:06d}.ply"), force="mesh", process=False)
            mesh.apply_scale(0.001)  # mm -> m, as the released LINEMOD reader
            sym_tfs = symmetry_tfs_from_info(info, rot_angle_discrete=5)
            if est is None:
                est = FoundationPose(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=sym_tfs,
                                     mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx, debug=0,
                                     debug_dir=str(ROOT / "bop6dof_results" / "_fp_debug"))
            else:
                est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=sym_tfs, mesh=mesh)
            cur_obj = (ds, obj)
        obs = np.load(fdir / "observation.npz")
        K = np.asarray(obs["K"], dtype=np.float64)
        rgb = np.ascontiguousarray(obs["rgb"])  # uint8 RGB
        depth_m = obs["depth"].astype(np.float32) / 1000.0
        mask = obs["mask"] > 0
        R_gt, t_gt = np.array(proto["gt_R"]), np.array(proto["gt_t"])
        tf_c = est.get_tf_to_centered_mesh().data.cpu().numpy()
        for init in inits:
            set_seed(0)
            try:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if args.mode == "full":
                    P = est.register(K=K, rgb=rgb, depth=depth_m, ob_mask=mask, iteration=args.iters)
                else:
                    T0 = np.eye(4)
                    T0[:3, :3] = np.array(init["R0"], dtype=np.float64)
                    T0[:3, 3] = np.array(init["t0"], dtype=np.float64) / 1000.0
                    Tc = T0 @ np.linalg.inv(tf_c)
                    if args.mode == "refine" and args.t0_translation == "fp":
                        Tc[:3, 3] = est.guess_translation(depth=depth_m, mask=mask, K=K)
                    est.pose_last = torch.as_tensor(Tc, device="cuda", dtype=torch.float)
                    P = est.track_one(rgb=rgb, depth=depth_m, K=K, iteration=args.iters)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                P = np.asarray(P, dtype=np.float64)
                R_est, t_est = P[:3, :3], P[:3, 3] * 1000.0
                met = compute_metrics(R_est, t_est, R_gt, t_gt, K, pts, info, syms)
                rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj,
                       "method": {"full": "foundationpose_full", "refine": "foundationpose_refine",
                                  "refine-from-ours": "ours_then_foundationpose"}[args.mode],
                       "bin": init["bin"], "init_rot_err_deg": init["rot_err_deg"], "R_est": R_est.tolist(), "t_est": t_est.tolist(),
                       "t_total_s": dt, **met}
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj,
                       "method": f"foundationpose_{args.mode}", "bin": init["bin"], "init_rot_err_deg": init["rot_err_deg"], "failed": True,
                       "error": str(e)[:200], "R_est": None, "t_est": None, "t_total_s": None, "e_R_deg": 180.0, "e_t_mm": float("inf"),
                       "add": float("inf"), "adds": float("inf"), "add_s_metric": float("inf"), "add_lt_0.1d": False, "mssd": float("inf"),
                       "mspd": float("inf"), "ar_mssd": 0.0, "ar_mspd": 0.0, "diameter": float(info["diameter"]), "symmetric": len(syms) > 1}
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{tag}] {args.mode}/{init['bin']}: e_R={rec['e_R_deg']:.1f} e_t={rec['e_t_mm']:.1f} ADD(-S)={rec['add_s_metric']:.1f} "
                  f"{rec['t_total_s'] or 0:.2f}s", flush=True)
    print(f"[done] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
