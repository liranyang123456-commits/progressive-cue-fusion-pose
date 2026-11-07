#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GoTrack (CVPRW 2025) refiner on the exported BOP frame packages, two modes:

  refine           : its refiner (5 iterations, the released test setting) from the protocol's T0 of every rotation bin.
  refine-from-ours : its refiner from the proposed method's estimate in results.jsonl.

Released checkpoint gotrack_checkpoint.pt (DINOv2 ViT-S/14-reg backbone, flow + confidence head, PnP-RANSAC on the
predicted 2D-3D correspondences, re-cropping every iteration) and released code, unchanged; RGB only. Run in WSL
Ubuntu (PyTorch 2.11 + CUDA 12.8, the released pyrender rasteriser headless through EGL):
  python baselines/run_gotrack_bop.py --root results/bop6dof_results/lmo --mode refine
Writes <root>/baseline_gotrack_<mode>.jsonl with the same metric fields as results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = Path(os.environ.get("GOTRACK_DIR", "external/gotrack"))
BOP_ROOT = Path(os.environ.get("BOP_ROOT", "data/bop"))
sys.path.insert(0, str(ROOT))
from bop_metrics import compute_metrics, eval_points_trimesh, load_models_info, symmetry_transforms  # noqa: E402

DATASETS = {
    "lm": {"root": BOP_ROOT / "lm", "models": "models", "models_eval": "models_eval"},
    "lmo": {"root": BOP_ROOT / "lmo", "models": "models", "models_eval": "models_eval"},
    "tless": {"root": BOP_ROOT / "tless", "models": "models_cad", "models_eval": "models_eval"},
}
OUT_NAME = {"refine": "baseline_gotrack_refine.jsonl", "refine-from-ours": "baseline_gotrack_from_ours.jsonl"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", default="refine", choices=list(OUT_NAME))
    ap.add_argument("--iters", type=int, default=5, help="refiner iterations (5 = the released test setting)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out-name", default="", help="override the output file name (e.g. for a sanity run)")
    ap.add_argument("--sanity-gt", action="store_true", help="refine from the ground-truth pose (pipeline check)")
    ap.add_argument("--ours-file", default="results.jsonl", help="refine-from-ours: file with the starting poses (R_est, t_est per tag)")
    args = ap.parse_args()

    os.chdir(GT_DIR)
    sys.path.insert(0, str(GT_DIR))
    import torch

    # The backbone is fetched with torch.hub from the DINOv2 main branch, which no longer matches the DINOv2
    # submodule pinned by GoTrack; serve the same hub entry point from the pinned submodule instead.
    hub_load = torch.hub.load

    def hub_load_pinned(repo_or_dir, model, *a, **kw):
        if repo_or_dir == "facebookresearch/dinov2":
            return hub_load(str(GT_DIR / "external" / "dinov2"), model, *a, source="local", **kw)
        return hub_load(repo_or_dir, model, *a, **kw)

    torch.hub.load = hub_load_pinned
    from bop_toolkit_lib import inout
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from utils import data_util, net_util, renderer_builder, structs

    root = Path(args.root)
    frames = sorted(d for d in root.iterdir() if d.is_dir() and (d / "protocol.json").exists())
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    out_path = root / (args.out_name or OUT_NAME[args.mode])
    ours_init = {}
    if args.mode == "refine-from-ours":
        for line in (root / args.ours_file).read_text(encoding="utf-8").splitlines():
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

    cfg = OmegaConf.load(GT_DIR / "configs" / "model" / "gotrack.yaml")
    cfg.opts.num_iterations_test = int(args.iters)
    cfg.opts.debug = False  # the default writes a visualisation per call, which would enter the timing
    model = instantiate(cfg)
    net_util.load_checkpoint(model=model, checkpoint_path=str(GT_DIR / "gotrack_checkpoint.pt"),
                             checkpoint_key="model_state_dict", prefix="models.1.")
    model = model.cuda().eval()
    tmp_dir = Path(tempfile.mkdtemp(prefix="gotrack_"))
    model.result_dir = tmp_dir
    cur_ds = None
    rng = np.random.default_rng(0)
    pts_cache, info_cache = {}, {}
    print(f"[gotrack:{args.mode}] {len(frames)} frames, iterations={args.iters}", flush=True)
    for fdir in frames:
        tag = fdir.name
        proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
        ds, obj = proto["dataset"], int(proto["obj_id"])
        if args.sanity_gt:
            inits = [{"bin": "gt", "R0": proto["gt_R"], "t0": proto["gt_t"], "rot_err_deg": 0.0}]
        elif args.mode == "refine":
            inits = proto["inits"]
        else:
            r0 = ours_init.get(tag)
            if r0 is None:
                print(f"[skip] {tag}: no estimate of the proposed method")
                continue
            inits = [{"bin": "ours", "R0": r0["R_est"], "t0": r0["t_est"], "rot_err_deg": float(r0["e_R_deg"])}]
        inits = [i for i in inits if (tag, str(i["bin"])) not in done]
        if not inits:
            continue
        cfg_ds = DATASETS[ds]
        if cur_ds != ds:
            model_tpath = str(cfg_ds["root"] / cfg_ds["models"]) + "/obj_{obj_id:06d}.ply"
            model.renderer = renderer_builder.build(renderer_type=renderer_builder.RendererType.PYRENDER_RASTERIZER,
                                                    model_path=model_tpath)
            model.object_vertices = {}
            cur_ds = ds
        if obj not in model.object_vertices:
            pts_ply = inout.load_ply(str(cfg_ds["root"] / cfg_ds["models"] / f"obj_{obj:06d}.ply"))["pts"]
            model.object_vertices[obj] = rng.permutation(pts_ply)[:1000].astype(np.float32)  # as the released BOP loader
        if (ds, obj) not in info_cache:
            pts_cache[(ds, obj)] = eval_points_trimesh(cfg_ds["root"] / cfg_ds["models_eval"], cfg_ds["root"] / cfg_ds["models"], obj)
            info_cache[(ds, obj)] = load_models_info(cfg_ds["root"] / cfg_ds["models"])[str(obj)]
        pts, info = pts_cache[(ds, obj)], info_cache[(ds, obj)]
        syms = symmetry_transforms(info)
        obs = np.load(fdir / "observation.npz")
        K = np.asarray(obs["K"], dtype=np.float64)
        rgb = np.ascontiguousarray(obs["rgb"])  # uint8 RGB
        H, W = rgb.shape[:2]
        R_gt, t_gt = np.array(proto["gt_R"]), np.array(proto["gt_t"])
        camera = structs.PinholePlaneCameraModel(width=W, height=H, f=(float(K[0, 0]), float(K[1, 1])),
                                                 c=(float(K[0, 2]), float(K[1, 2])), T_world_from_eye=np.eye(4, dtype=np.float32))
        for init in inits:
            try:
                # Linemod ground-truth rotations, and the T0 derived from them, deviate from orthonormality by up to
                # 3e-4, above the 1e-5 rigidity check of GoTrack's camera model: project onto SO(3) first.
                U, _, Vt = np.linalg.svd(np.array(init["R0"], dtype=np.float64))
                R0 = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
                t0 = np.array(init["t0"], dtype=np.float64).reshape(3, 1)  # mm, as the released BOP loader
                anno = structs.ObjectAnnotation(dataset=ds, lid=obj, pose=structs.ObjectPose(R=R0, t=t0))
                so = structs.SceneObservation(scene_id=int(proto["scene_id"]), im_id=int(proto["im_id"]), image=rgb,
                                              camera=camera, objects_anno=[anno], time=0.0)
                inputs = data_util.convert_list_scene_observations_to_gotrack_inputs([{"scene_observation": so, "target_objects": None}])
                torch.cuda.synchronize()
                ts = time.perf_counter()
                with torch.no_grad():
                    out = model.forward_pipeline(inputs, batch_idx=0)
                torch.cuda.synchronize()
                dt = time.perf_counter() - ts
                P = out["objects"].poses_cam_from_model[0].cpu().numpy().astype(np.float64)
                R_est, t_est = P[:3, :3], P[:3, 3]
                met = compute_metrics(R_est, t_est, R_gt, t_gt, K, pts, info, syms)
                rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj,
                       "method": {"refine": "gotrack_refine", "refine-from-ours": "ours_then_gotrack"}[args.mode],
                       "bin": init["bin"], "init_rot_err_deg": init["rot_err_deg"], "R_est": R_est.tolist(), "t_est": t_est.tolist(),
                       "pose_score": float(out["objects"].pose_scores[0]), "t_total_s": dt, **met}
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj,
                       "method": f"gotrack_{args.mode}", "bin": init["bin"], "init_rot_err_deg": init["rot_err_deg"], "failed": True,
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
