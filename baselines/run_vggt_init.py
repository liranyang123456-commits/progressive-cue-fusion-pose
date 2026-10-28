#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foundation-model initialisation experiment: VGGT (CVPR 2025) as an *initialiser* for a known mesh.

VGGT takes a set of RGB images and predicts, in one pass, the extrinsics of every image in the frame
of the first image (up to a global scale) and point maps. It uses neither the mesh nor an initial
pose, so it cannot return an object pose by itself. We give it the real crop (image 0) together with
12 rendered templates of the known mesh whose object->camera poses are known. The predicted template
camera centres are related to the known ones by a similarity transform (R, s, t); Umeyama alignment
recovers it, and because image 0 is VGGT's world origin, (R, t/s) is the object->real-camera pose.
The resulting pose is written as an additional initial-pose "bin" (vggt) for the refiners and
evaluated on its own with the shared metrics.

Run in the `wapr` env (pip install git+https://github.com/facebookresearch/vggt):
  python baselines/run_vggt_init.py --root bop6dof_results/lmo
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
from bop_metrics import compute_metrics, eval_points_trimesh, load_models_info, symmetry_transforms  # noqa: E402

DATASETS = {
    "lm": {"root": Path(os.environ.get("BOP_ROOT", "data/bop")) / "lm", "models": "models", "models_eval": "models_eval"},
    "lmo": {"root": Path(os.environ.get("BOP_ROOT", "data/bop")) / "lmo", "models": "models", "models_eval": "models_eval"},
    "tless": {"root": Path(os.environ.get("BOP_ROOT", "data/bop")) / "tless", "models": "models_cad", "models_eval": "models_eval"},
}


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Similarity (R, s, t) with dst ~= s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    cov = Y.T @ X / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (X ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / max(var_s, 1e-12))
    t = mu_d - s * R @ mu_s
    return R, s, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    from PIL import Image
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    root = Path(args.root)
    frames = sorted([d for d in root.iterdir() if d.is_dir() and (d / "fm_templates" / "meta.json").exists()])
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    out_path = root / "baseline_vggt_init.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["tag"])
            except Exception:
                pass
    pts_cache, info_cache = {}, {}
    for fdir in frames:
        tag = fdir.name
        if tag in done:
            continue
        proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
        meta = json.loads((fdir / "fm_templates" / "meta.json").read_text(encoding="utf-8"))
        ds, obj = proto["dataset"], int(proto["obj_id"])
        cfg = DATASETS[ds]
        if obj not in pts_cache:
            pts_cache[obj] = eval_points_trimesh(cfg["root"] / cfg["models_eval"], cfg["root"] / cfg["models"], obj)
            info_cache[obj] = load_models_info(cfg["root"] / cfg["models"])[str(obj)]
        pts, info = pts_cache[obj], info_cache[obj]
        paths = [str(fdir / "fm_templates" / "real_crop.png")] + [str(fdir / "fm_templates" / t["file"]) for t in meta["templates"]]
        t0 = time.perf_counter()
        images = load_and_preprocess_images(paths).to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=dtype):
                aggregated_tokens_list, ps_idx = model.aggregator(images[None])
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        E = extrinsic[0].float().cpu().numpy()  # (N, 3, 4) camera-from-world, world = image 0
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        # known template cameras in the object frame
        c_obj, c_pred, R_est_list = [], [], []
        for i, tpl in enumerate(meta["templates"], start=1):
            R_p, t_p = np.array(tpl["R"]), np.array(tpl["t"])
            R_i, t_i = E[i][:3, :3], E[i][:3, 3]
            c_obj.append(-R_p.T @ t_p)
            c_pred.append(-R_i.T @ t_i)
            R_est_list.append(R_i.T @ R_p)  # object -> world (= real camera) rotation estimate from template i
        c_obj, c_pred = np.array(c_obj), np.array(c_pred)
        R_sim, s, t_sim = umeyama(c_obj, c_pred)
        # chordal mean of the per-template rotation estimates as a consistency check / alternative
        M = np.mean(np.stack(R_est_list), axis=0)
        U, _, Vt = np.linalg.svd(M)
        R_chordal = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        R0 = R_sim
        t0_mm = t_sim / max(s, 1e-9)
        # residual of the similarity fit (in predicted units, relative)
        resid = float(np.linalg.norm((s * (c_obj @ R_sim.T) + t_sim) - c_pred, axis=1).mean() / (np.linalg.norm(c_pred - c_pred.mean(0), axis=1).mean() + 1e-9))
        R_gt, t_gt = np.array(proto["gt_R"]), np.array(proto["gt_t"])
        met = compute_metrics(R0, t0_mm, R_gt, t_gt, np.load(fdir / "observation.npz")["K"], pts, info, symmetry_transforms(info))
        rot_consistency = float(np.degrees(np.arccos(np.clip((np.trace(R_chordal @ R_sim.T) - 1) / 2, -1, 1))))
        rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj, "method": "vggt_init",
               "R_est": R0.tolist(), "t_est": t0_mm.tolist(), "scale": s, "fit_residual_rel": resid, "rot_consistency_deg": rot_consistency,
               "t_total_s": dt, **met}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        # also append as an extra initial pose bin in the protocol for the refiners
        proto.setdefault("inits_fm", [])
        proto["inits_fm"] = [x for x in proto["inits_fm"] if x.get("bin") != "vggt"] + [{"bin": "vggt", "R0": R0.tolist(), "t0": t0_mm.tolist(), "rot_err_deg": met["e_R_deg"]}]
        (fdir / "protocol.json").write_text(json.dumps(proto, indent=2), encoding="utf-8")
        print(f"[{tag}] VGGT init: e_R={met['e_R_deg']:.1f} e_t={met['e_t_mm']:.0f} mm  fit_res={resid:.2f} rot_cons={rot_consistency:.1f} ({dt:.1f}s)", flush=True)
    print(f"[done] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
