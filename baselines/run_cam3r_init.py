#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foundation-model initialisation experiment, fourth model: CAM3R (ECCV 2026) as an *initialiser* for a known mesh.

CAM3R's public release is its two-view network (the multi-view Ray-Aware Global Alignment is not released), so the
real RGB crop is paired with each of the 12 rendered templates of the known mesh (tools/render_templates_for_fm.py),
images prepared as in the released zero-shot script (512 x 512 for square inputs, ImgNorm, placeholder rays). For
pair i CAM3R returns T_i, the transform from template camera i to the real camera, in its own (unknown) unit; with
the known object -> template pose (R_i, t_i) this gives one object -> real-camera estimate
    R^(i) = R_Ti R_i,   t^(i) = R_Ti t_i + s_i t_Ti,
where s_i converts CAM3R's unit to mm: the median depth of the front half of the object's evaluation points in
template camera i divided by the median predicted depth on the template's object pixels (template geometry only;
the real depth is not used, as for VGGT / pi^3 / Wid3R). The estimate is that of the pair with the highest mean
predicted confidence on the template's object pixels (template retrieval by the network's own confidence); a
rotation-consensus fusion of the 12 estimates (inliers within 15 deg of the estimate with most agreement) and the
best of the 12 pair estimates (ground truth, diagnostic only) are recorded as well. The estimate is written as an
extra initial-pose "bin" (cam3r) for the refiners.

Run in the `cam3r` env (external/cam3r cloned, checkpoints/cam3r.pth downloaded), with PYTHONIOENCODING=utf-8:
  python baselines/run_cam3r_init.py --root bop6dof_results/lmo
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
CAM3R = ROOT / "external" / "cam3r"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CAM3R))
from bop_metrics import compute_metrics, eval_points_trimesh, load_models_info, symmetry_transforms  # noqa: E402

DATASETS = {
    "lm": {"root": Path(os.environ.get("BOP_ROOT", "data/bop")) / "lm", "models": "models", "models_eval": "models_eval"},
    "lmo": {"root": Path(os.environ.get("BOP_ROOT", "data/bop")) / "lmo", "models": "models", "models_eval": "models_eval"},
    "tless": {"root": Path(os.environ.get("BOP_ROOT", "data/bop")) / "tless", "models": "models_cad", "models_eval": "models_eval"},
}
INLIER_DEG = 15.0


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1.0) / 2.0, -1.0, 1.0))))


def chordal_mean(Rs: list[np.ndarray]) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.mean(np.stack(Rs), axis=0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def build_model(device):
    """The released network configuration of eval_two_view_zeroshot.build_model; the checkpoint stores the training
    arguments as an argparse.Namespace, which torch>=2.6 must be told to accept."""
    import argparse as _ap
    import torch
    from dust3r.model_with_unik3d_encoder_dust3r_adapter_relpose import AsymmetricCroCo3DStereo, inf

    model = AsymmetricCroCo3DStereo(
        pos_embed="RoPE100", patch_embed_cls="ManyAR_PatchEmbed", img_size=(512, 384), head_type="dpt",
        output_mode="pts3d", depth_mode=("softplus", 0, inf), conf_mode=("exp", 1, inf), enc_embed_dim=1024,
        enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12,
        adapter_config=str(CAM3R / "dust3r" / "vitl.json"), adapter_pretrained=True, adapter_use_layernorm_pool=True)
    torch.serialization.add_safe_globals([_ap.Namespace])
    ckpt = torch.load(CAM3R / "checkpoints" / "cam3r.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    return model.to(device).eval()


def load_view(path: Path, idx: int):
    """As eval_two_view_zeroshot.load_and_resize_image + load_images_from_folder for a square image."""
    import cv2
    import PIL.Image
    import torch
    from PIL.ImageOps import exif_transpose
    from dust3r.utils.image import ImgNorm

    img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    pil = exif_transpose(PIL.Image.fromarray(img)).convert("RGB")
    tw, th = (512, 512) if pil.size[0] == pil.size[1] else (512, 384)
    arr = cv2.resize(np.array(pil), (tw, th), interpolation=cv2.INTER_AREA)
    view = dict(img=ImgNorm(PIL.Image.fromarray(arr))[None], true_shape=np.int32([[th, tw]]), idx=idx, instance=path.stem)
    view["gt_rays"] = torch.zeros(1, th, tw, 3, dtype=torch.float32)
    return view, arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    from dust3r.inference import inference

    device = torch.device(args.device)
    model = build_model(device)
    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    frames = sorted([d for d in root.iterdir() if d.is_dir() and (d / "fm_templates" / "meta.json").exists()])
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    out_path = root / "baseline_cam3r_init.jsonl"
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
        t0 = time.perf_counter()
        real, _ = load_view(fdir / "fm_templates" / "real_crop.png", 0)
        pairs, masks = [], []
        for k, tpl in enumerate(meta["templates"], start=1):
            v, arr = load_view(fdir / "fm_templates" / tpl["file"], k)
            pairs.append((real, v))
            masks.append(arr.max(axis=-1) > 8)  # templates are rendered on a black background
        with torch.no_grad():
            res = inference(pairs, model, device, batch_size=4, verbose=False, demo=True)
        T = res["pred2"]["T"].float().cpu().numpy()          # (12, 4, 4) template camera -> real camera
        P2 = res["pred2"]["pts3d"].float().cpu().numpy()     # (12, H, W, 3) template-camera point maps
        C2 = res["pred2"]["conf"].float().cpu().numpy()      # (12, H, W) confidence of the template-view predictions
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        Rs, ts, scales, confs = [], [], [], []
        for k, tpl in enumerate(meta["templates"]):
            R_i, t_i = np.array(tpl["R"]), np.array(tpl["t"])
            z_obj = (pts @ R_i.T + t_i)[:, 2]
            z_true = float(np.median(z_obj[z_obj <= np.median(z_obj)]))
            m = masks[k] & np.isfinite(P2[k][..., 2]) & (P2[k][..., 2] > 0)
            z_pred = float(np.median(P2[k][..., 2][m])) if m.any() else float("nan")
            s_k = z_true / z_pred if z_pred > 0 else float("nan")
            Rs.append(T[k][:3, :3] @ R_i)
            ts.append(T[k][:3, :3] @ t_i + s_k * T[k][:3, 3])
            scales.append(s_k)
            confs.append(float(np.mean(C2[k][masks[k]])) if masks[k].any() else float("-inf"))
        counts = [sum(rot_angle_deg(Ra, Rb) <= INLIER_DEG for Rb in Rs) for Ra in Rs]
        best = int(np.argmax(counts))
        inl = [j for j in range(len(Rs)) if rot_angle_deg(Rs[best], Rs[j]) <= INLIER_DEG and np.all(np.isfinite(ts[j]))]
        R_cons = chordal_mean([Rs[j] for j in inl])
        spread = float(np.median([rot_angle_deg(R_cons, R) for R in Rs]))
        k_best = int(np.argmax(confs))
        R0, t0_mm = Rs[k_best], ts[k_best]
        R_gt, t_gt = np.array(proto["gt_R"]), np.array(proto["gt_t"])
        e_cons = rot_angle_deg(R_cons, R_gt)
        e_pairs = [rot_angle_deg(R, R_gt) for R in Rs]
        met = compute_metrics(R0, t0_mm, R_gt, t_gt, np.load(fdir / "observation.npz")["K"], pts, info, symmetry_transforms(info))
        rec = {"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj, "method": "cam3r_init",
               "R_est": R0.tolist(), "t_est": t0_mm.tolist(), "selected_template": k_best, "conf_selected": confs[k_best],
               "e_R_consensus_deg": e_cons, "n_inliers": len(inl), "spread_deg": spread,
               "e_R_best_pair_deg_diag": float(min(e_pairs)), "e_R_pairs_deg_diag": e_pairs,
               "scale_median": float(np.nanmedian(scales)), "t_total_s": dt, **met}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        proto.setdefault("inits_fm", [])
        proto["inits_fm"] = [x for x in proto["inits_fm"] if x.get("bin") != "cam3r"] + [{"bin": "cam3r", "R0": R0.tolist(), "t0": t0_mm.tolist(), "rot_err_deg": met["e_R_deg"]}]
        (fdir / "protocol.json").write_text(json.dumps(proto, indent=2), encoding="utf-8")
        print(f"[{tag}] cam3r init: e_R={met['e_R_deg']:.1f} e_t={met['e_t_mm']:.0f} mm (template {k_best})  consensus e_R={e_cons:.1f} "
              f"inliers={len(inl)}/12  best pair {min(e_pairs):.1f}  ({dt:.1f}s)", flush=True)
    print(f"[done] {out_path}")
    return 0


if __name__ == "__main__":
    os.chdir(CAM3R)
    raise SystemExit(main())
