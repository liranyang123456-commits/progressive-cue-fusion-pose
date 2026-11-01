#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foundation-model initialisation experiment, second model: pi^3 (ICLR 2026) as an *initialiser* for a known mesh.

Same protocol as baselines/run_vggt_init.py: the real RGB crop (image 0) is fed together with the 12 rendered
templates of the known mesh (tools/render_templates_for_fm.py). pi^3 returns camera-to-world matrices in a
frame of its own (it is permutation-equivariant, so no image is privileged); the predicted template camera
centres are related to the known object-frame centres by a similarity (R, s, t) recovered by Umeyama alignment,
and the pose of the real camera in that frame gives the object -> real-camera pose:
    X_cam0 = W2C_0 (s R X_obj + t)   =>   R_est = R0w R,  t_est = (R0w t + t0w) / s   (object units, mm).
The estimate is written as an extra initial-pose "bin" (pi3) for the refiners and evaluated on its own.

Run in the `wapr` env (needs external/Pi3 cloned: git clone https://github.com/yyfz/Pi3 external/Pi3):
  python baselines/run_pi3_init.py --root bop6dof_results/lmo
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
sys.path.insert(0, str(ROOT / "external" / "Pi3"))
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


def load_images(paths: list[str], pixel_limit: int = 255000):
    """pi^3's loader (pi3.utils.basic.load_images_as_tensor) for an explicit list of files: uniform size,
    multiples of 14, at most `pixel_limit` pixels, values in [0, 1]."""
    import torch
    from PIL import Image

    ims = [Image.open(p).convert("RGB") for p in paths]
    W0, H0 = ims[0].size
    scale = math.sqrt(pixel_limit / (W0 * H0))
    Wt, Ht = W0 * scale, H0 * scale
    k, m = round(Wt / 14), round(Ht / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > Wt / Ht:
            k -= 1
        else:
            m -= 1
    TW, TH = max(1, k) * 14, max(1, m) * 14
    arr = [np.asarray(im.resize((TW, TH), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0 for im in ims]
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="", help="local model.safetensors (default: download yyfz233/Pi3 from the Hub)")
    args = ap.parse_args()
    import torch
    from pi3.models.pi3 import Pi3

    device = torch.device(args.device)
    # pi^3 routes bfloat16 through the flash-attention SDPA kernel only, which this torch build does not provide for sm_120;
    # float16 uses the math / memory-efficient kernels (inference only, no observable change in the estimates)
    dtype = torch.float16
    if args.ckpt:
        from safetensors.torch import load_file
        model = Pi3().to(device).eval()
        model.load_state_dict(load_file(args.ckpt))
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    root = Path(args.root)
    frames = sorted([d for d in root.iterdir() if d.is_dir() and (d / "fm_templates" / "meta.json").exists()])
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    out_path = root / "baseline_pi3_init.jsonl"
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
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                res = model(imgs[None])
        P = res["camera_poses"][0].float().cpu().numpy()  # (N, 4, 4) camera-to-world, OpenCV convention
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        # similarity between the known template camera centres (object frame, mm) and the predicted ones (pi^3 frame)
        c_obj, c_pred, R_obj2w = [], [], []
        for i, tpl in enumerate(meta["templates"], start=1):
            R_p, t_p = np.array(tpl["R"]), np.array(tpl["t"])  # object -> template camera i
            c_obj.append(-R_p.T @ t_p)
            c_pred.append(P[i][:3, 3])
            R_obj2w.append(P[i][:3, :3] @ R_p)  # object -> pi^3 world via template i
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
        rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj, "method": "pi3_init",
               "R_est": R0.tolist(), "t_est": t0_mm.tolist(), "scale": s, "fit_residual_rel": resid, "rot_consistency_deg": rot_consistency,
               "t_total_s": dt, **met}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        proto.setdefault("inits_fm", [])
        proto["inits_fm"] = [x for x in proto["inits_fm"] if x.get("bin") != "pi3"] + [{"bin": "pi3", "R0": R0.tolist(), "t0": t0_mm.tolist(), "rot_err_deg": met["e_R_deg"]}]
        (fdir / "protocol.json").write_text(json.dumps(proto, indent=2), encoding="utf-8")
        print(f"[{tag}] pi3 init: e_R={met['e_R_deg']:.1f} e_t={met['e_t_mm']:.0f} mm  fit_res={resid:.2f} rot_cons={rot_consistency:.1f} ({dt:.1f}s)", flush=True)
    print(f"[done] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
