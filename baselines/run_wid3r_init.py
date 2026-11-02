#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foundation-model initialisation experiment, third model: Wid3R (ECCV 2026) as an *initialiser* for a known mesh.

Same protocol as baselines/run_pi3_init.py (Wid3R is fine-tuned from pi^3 and returns camera-to-world matrices in
the same convention): the real RGB crop (image 0) and the 12 rendered templates of the known mesh are fed together;
the predicted template camera centres are aligned to the known object-frame centres by a similarity (Umeyama), and
the pose of the real camera in that frame gives the object -> real-camera pose. Images and cameras are prepared as in
the released evaluation code: resized to 518 x 336 and passed as pinhole cameras with placeholder parameters
(camera-model token only, no intrinsics: the rays are predicted).

Run in the `wapr` env (needs external/Wid3R cloned and pretrained_weights/wid3r.bin downloaded):
  python baselines/run_wid3r_init.py --root bop6dof_results/lmo
"""
from __future__ import annotations
import os

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "Wid3R"))
import types  # noqa: E402
sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))  # imported but unused by the released Wid3R modules
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


def load_images(paths: list[str], width: int = 518, height: int = 336):
    """Wid3R's evaluation loader (evaluation/utils/interfaces.load_image_dk): every image is resized to the single
    training / evaluation resolution 518 x 336 (LANCZOS, no cropping), values in [0, 1]."""
    import torch
    from PIL import Image

    arr = [np.asarray(Image.open(p).convert("RGB").resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
           for p in paths]
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default=str(ROOT / "external" / "Wid3R" / "pretrained_weights" / "wid3r.bin"))
    args = ap.parse_args()
    import torch
    from wid3r.models.wid3r_training import Wid3R
    from cam_utils.camera import Pinhole

    device = torch.device(args.device)
    # as for pi^3: bfloat16 is routed through the flash-attention SDPA kernel only, which this torch build does not
    # provide for sm_120; float16 uses the math / memory-efficient kernels
    dtype = torch.float16
    model = Wid3R(pos_type="rope100", decoder_size="large", load_vggt=False, freeze_encoder=True, use_global_points=False,
                  train_conf=False, num_dec_blk_not_to_checkpoint=4, ckpt=None, use_camera_gt=False)
    model.load_state_dict(torch.load(args.ckpt, weights_only=False, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    root = Path(args.root)
    frames = sorted([d for d in root.iterdir() if d.is_dir() and (d / "fm_templates" / "meta.json").exists()])
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    out_path = root / "baseline_wid3r_init.jsonl"
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
        imgs = load_images(paths).to(device)
        n_img, _, h_img, w_img = imgs.shape
        cameras = torch.cat([Pinhole(params=torch.from_numpy(np.array([1, 1, w_img, h_img]))) for _ in range(n_img)]).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                res = model(imgs[None], cameras=cameras)
        P = res["camera_poses"][0].float().cpu().numpy()  # (N, 4, 4) camera-to-world, OpenCV convention
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        # similarity between the known template camera centres (object frame, mm) and the predicted ones (Wid3R frame)
        c_obj, c_pred, R_obj2w = [], [], []
        for i, tpl in enumerate(meta["templates"], start=1):
            R_p, t_p = np.array(tpl["R"]), np.array(tpl["t"])  # object -> template camera i
            c_obj.append(-R_p.T @ t_p)
            c_pred.append(P[i][:3, 3])
            R_obj2w.append(P[i][:3, :3] @ R_p)  # object -> Wid3R world via template i
        c_obj, c_pred = np.array(c_obj), np.array(c_pred)
        R_sim, s, t_sim = umeyama(c_obj, c_pred)
        # real camera: world -> camera 0
        R0w = P[0][:3, :3].T
        t0w = -R0w @ P[0][:3, 3]
        R0 = R0w @ R_sim
        t0_mm = (R0w @ t_sim + t0w) / max(s, 1e-9)
        # chordal mean of the per-template rotation estimates (consistency check)
        M = np.mean(np.stack(R_obj2w), axis=0)
        U, _, Vt = np.linalg.svd(M)
        R_chordal_w = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        rot_consistency = float(np.degrees(np.arccos(np.clip((np.trace((R0w @ R_chordal_w) @ R0.T) - 1) / 2, -1, 1))))
        resid = float(np.linalg.norm((s * (c_obj @ R_sim.T) + t_sim) - c_pred, axis=1).mean() / (np.linalg.norm(c_pred - c_pred.mean(0), axis=1).mean() + 1e-9))
        R_gt, t_gt = np.array(proto["gt_R"]), np.array(proto["gt_t"])
        met = compute_metrics(R0, t0_mm, R_gt, t_gt, np.load(fdir / "observation.npz")["K"], pts, info, symmetry_transforms(info))
        rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj, "method": "wid3r_init",
               "R_est": R0.tolist(), "t_est": t0_mm.tolist(), "scale": s, "fit_residual_rel": resid, "rot_consistency_deg": rot_consistency,
               "t_total_s": dt, **met}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        proto.setdefault("inits_fm", [])
        proto["inits_fm"] = [x for x in proto["inits_fm"] if x.get("bin") != "wid3r"] + [{"bin": "wid3r", "R0": R0.tolist(), "t0": t0_mm.tolist(), "rot_err_deg": met["e_R_deg"]}]
        (fdir / "protocol.json").write_text(json.dumps(proto, indent=2), encoding="utf-8")
        print(f"[{tag}] wid3r init: e_R={met['e_R_deg']:.1f} e_t={met['e_t_mm']:.0f} mm  fit_res={resid:.2f} rot_cons={rot_consistency:.1f} ({dt:.1f}s)", flush=True)
    print(f"[done] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
