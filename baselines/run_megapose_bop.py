#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MegaPose (CoRL 2022, HappyPose implementation) on the exported BOP frame packages, three modes:

  full             : its own estimator from the 2D box of the visible mask (coarse network scoring the 576-rotation SO(3)
                     grid placed in the box, best hypothesis refined for 5 iterations) -- initialisation-free.
  refine           : its refiner (5 iterations) from the protocol's T0 of every rotation bin.
  refine-from-ours : its refiner from the proposed method's estimate in results.jsonl.

Released networks (coarse-rgb-906902141, refiner-rgb-653307694 / refiner-rgbd-288182519) through the named models
megapose-1.0-RGB / megapose-1.0-RGBD of HappyPose; released code unchanged. Run in WSL Ubuntu (PyTorch 2.11 + CUDA
12.8, Panda3D renderer through WSLg):
  python baselines/run_megapose_bop.py --root results/bop6dof_results/lmo --mode refine
Writes <root>/baseline_megapose_<mode>[_rgb].jsonl with the same metric fields as results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("HAPPYPOSE_DATA_DIR", str(Path.home() / "happypose_data"))
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BOP_ROOT = Path(os.environ.get("BOP_ROOT", "data/bop"))
sys.path.insert(0, str(ROOT))
from bop_metrics import compute_metrics, eval_points_trimesh, load_models_info, symmetry_transforms  # noqa: E402

DATASETS = {
    "lm": {"root": BOP_ROOT / "lm", "models": "models", "models_eval": "models_eval"},
    "lmo": {"root": BOP_ROOT / "lmo", "models": "models", "models_eval": "models_eval"},
    "tless": {"root": BOP_ROOT / "tless", "models": "models_cad", "models_eval": "models_eval"},
}
OUT_NAME = {"full": "baseline_megapose_full", "refine": "baseline_megapose_refine", "refine-from-ours": "baseline_megapose_from_ours"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", default="refine", choices=list(OUT_NAME))
    ap.add_argument("--model", default="megapose-1.0-RGBD", choices=["megapose-1.0-RGBD", "megapose-1.0-RGB"])
    ap.add_argument("--iters", type=int, default=5, help="refiner iterations (5 = the released setting)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out-name", default="", help="override the output file name (e.g. for a sanity run)")
    ap.add_argument("--sanity-gt", action="store_true", help="refine from the ground-truth pose (pipeline check)")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
    from happypose.toolbox.inference.types import ObservationTensor
    from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model
    from happypose.toolbox.utils.tensor_collection import PandasTensorCollection

    root = Path(args.root)
    frames = sorted(d for d in root.iterdir() if d.is_dir() and (d / "protocol.json").exists())
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    suffix = "" if args.model == "megapose-1.0-RGBD" else "_rgb"
    out_path = root / (args.out_name or f"{OUT_NAME[args.mode]}{suffix}.jsonl")
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

    protos = {f.name: json.loads((f / "protocol.json").read_text(encoding="utf-8")) for f in frames}
    ds_all = sorted({p["dataset"] for p in protos.values()})
    assert len(ds_all) == 1, ds_all
    ds = ds_all[0]
    cfg = DATASETS[ds]
    obj_ids = sorted({int(p["obj_id"]) for p in protos.values()})
    object_dataset = RigidObjectDataset([RigidObject(label=f"obj_{o:06d}", mesh_path=cfg["root"] / cfg["models"] / f"obj_{o:06d}.ply",
                                                     mesh_units="mm") for o in obj_ids])
    est = load_named_model(args.model, object_dataset, n_workers=1).cuda()
    params = dict(NAMED_MODELS[args.model]["inference_parameters"])
    params["n_refiner_iterations"] = int(args.iters)
    requires_depth = bool(NAMED_MODELS[args.model]["requires_depth"])
    pts_cache, info_cache = {}, {}
    print(f"[megapose:{args.mode}:{args.model}] {len(frames)} frames ({ds}), objects {obj_ids}, iterations={args.iters}", flush=True)
    for fdir in frames:
        tag = fdir.name
        proto = protos[tag]
        obj = int(proto["obj_id"])
        if args.sanity_gt:
            inits = [{"bin": "gt", "R0": proto["gt_R"], "t0": proto["gt_t"], "rot_err_deg": 0.0}]
        elif args.mode == "full":
            inits = [{"bin": "none", "rot_err_deg": None}]
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
        if (ds, obj) not in info_cache:
            pts_cache[(ds, obj)] = eval_points_trimesh(cfg["root"] / cfg["models_eval"], cfg["root"] / cfg["models"], obj)
            info_cache[(ds, obj)] = load_models_info(cfg["root"] / cfg["models"])[str(obj)]
        pts, info = pts_cache[(ds, obj)], info_cache[(ds, obj)]
        syms = symmetry_transforms(info)
        obs = np.load(fdir / "observation.npz")
        K = np.asarray(obs["K"], dtype=np.float64)
        rgb = np.ascontiguousarray(obs["rgb"])  # uint8 RGB
        depth_m = obs["depth"].astype(np.float32) / 1000.0 if requires_depth else None
        mask = obs["mask"] > 0
        R_gt, t_gt = np.array(proto["gt_R"]), np.array(proto["gt_t"])
        observation = ObservationTensor.from_numpy(rgb, depth_m, K.astype(np.float32)).cuda()
        infos = pd.DataFrame({"label": [f"obj_{obj:06d}"], "batch_im_id": [0], "instance_id": [0]})
        for init in inits:
            try:
                torch.cuda.synchronize()
                ts = time.perf_counter()
                if args.mode == "full" and not args.sanity_gt:
                    ys, xs = np.nonzero(mask)
                    box = torch.as_tensor([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=torch.float32)
                    det = PandasTensorCollection(infos=infos.copy(), bboxes=box).cuda()
                    out, _ = est.run_inference_pipeline(observation, detections=det, run_detector=False, **params)
                    P = out.poses[0].cpu().numpy().astype(np.float64)
                else:
                    T0 = np.eye(4, dtype=np.float32)
                    T0[:3, :3] = np.array(init["R0"], dtype=np.float32)
                    T0[:3, 3] = np.array(init["t0"], dtype=np.float32) / 1000.0
                    data = PandasTensorCollection(infos=infos.copy(), poses=torch.as_tensor(T0[None])).cuda()
                    preds, _ = est.forward_refiner(observation, data, n_iterations=int(args.iters))
                    P = preds[f"iteration={int(args.iters)}"].poses[0].cpu().numpy().astype(np.float64)
                torch.cuda.synchronize()
                dt = time.perf_counter() - ts
                R_est, t_est = P[:3, :3], P[:3, 3] * 1000.0
                met = compute_metrics(R_est, t_est, R_gt, t_gt, K, pts, info, syms)
                rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj,
                       "method": {"full": "megapose_full", "refine": "megapose_refine", "refine-from-ours": "ours_then_megapose"}[args.mode],
                       "model": args.model, "bin": init["bin"], "init_rot_err_deg": init["rot_err_deg"], "R_est": R_est.tolist(),
                       "t_est": t_est.tolist(), "t_total_s": dt, **met}
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj,
                       "method": f"megapose_{args.mode}", "model": args.model, "bin": init["bin"], "init_rot_err_deg": init["rot_err_deg"],
                       "failed": True, "error": str(e)[:200], "R_est": None, "t_est": None, "t_total_s": None, "e_R_deg": 180.0,
                       "e_t_mm": float("inf"), "add": float("inf"), "adds": float("inf"), "add_s_metric": float("inf"), "add_lt_0.1d": False,
                       "mssd": float("inf"), "mspd": float("inf"), "ar_mssd": 0.0, "ar_mspd": 0.0, "diameter": float(info["diameter"]),
                       "symmetric": len(syms) > 1}
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{tag}] {args.mode}/{init['bin']}: e_R={rec['e_R_deg']:.1f} e_t={rec['e_t_mm']:.1f} ADD(-S)={rec['add_s_metric']:.1f} "
                  f"{rec['t_total_s'] or 0:.2f}s", flush=True)
    print(f"[done] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
