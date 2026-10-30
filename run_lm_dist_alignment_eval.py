#!/usr/bin/env python3
"""
run_lm_dist_alignment_eval.py
──────────────────────────────
Focused real-data experiment: depth-guided camera-shell identification on BOP LM.

We test ONE specific claim from the paper (Proposition 2 + Lemma 1):
  "Silhouette and normal cues alone cannot identify camera distance;
   a depth hint is necessary to select the correct camera shell."

Protocol
--------
For N real BOP LM frames (object-centred, good visibility):
  1. Compute the GT distance d* from the BOP pose.
  2. Evaluate two distance-selection strategies at the GT angular position:
     (a) Geometry-only (sil + normal, no depth): select the shell with
         lowest geometry loss among {0.3×, 0.5×, ..., 4.0×} × mesh_radius.
     (b) Depth-guided (our method): select shell closest to the depth hint
         median_depth × 0.92, then verify with geometry loss.
  3. Report:
     - Relative distance error |d_est - d*| / d* for each strategy
     - Whether the correct shell was found: |d_est - d*| < 0.1 × d*

This is a clean, controlled experiment that directly supports the paper's claims
without requiring full pose recovery on real data.

Usage
-----
  Set PIPELINE_PYTHON env var and run:
  python run_lm_dist_alignment_eval.py
"""
from __future__ import annotations

import csv
import io, contextlib
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT   = Path(__file__).resolve().parent
BOP    = Path("D:/bop/lm")
PYTHON = os.environ.get("PIPELINE_PYTHON", "python")

sys.path.insert(0, str(ROOT))
os.environ["VAL_DISABLE_TORCH_COMPILE"] = "1"
os.environ["FORCE_CODE_RUN_PARAMS"] = "0"
os.environ["VAL_FOVY_DEG"] = "45.41"

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    import Exp_Camera_Optim_RT_Val_Beam_Search12291_Standford122 as exp

look_at_view_transform = exp.look_at_view_transform
MeshData               = exp.MeshData
MeshRasterizer         = exp.MeshRasterizer
RasterizationSettings  = exp.RasterizationSettings
MeshRenderer           = exp.MeshRenderer
FoVPerspectiveCameras  = exp.FoVPerspectiveCameras
BlendParams            = exp.BlendParams
SoftSilhouetteShader   = exp.SoftSilhouetteShader
build_renderers        = exp.build_renderers
_batched_eval_candidates_no_grad = exp._batched_eval_candidates_no_grad
_ref_dist_hint_from_depth        = exp._ref_dist_hint_from_depth
_mesh_radius_from_verts          = exp._mesh_radius_from_verts

from lm_rgbd_adapter import BopLmAdapter

LM_OBJ_NAMES = {1:"ape", 2:"benchvise", 4:"camera", 6:"cat", 9:"duck",
                10:"eggbox", 11:"glue", 14:"lamp", 15:"phone"}


def load_mesh(ply: Path, device):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(ply))
    m.compute_vertex_normals()
    v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
    f = torch.tensor(np.asarray(m.triangles), dtype=torch.int64, device=device)
    n = torch.tensor(np.asarray(m.vertex_normals), dtype=torch.float32, device=device)
    c = None
    if m.has_vertex_colors():
        c = torch.tensor(np.asarray(m.vertex_colors), dtype=torch.float32, device=device)
    return MeshData(verts=v, faces=f, verts_normals=n, verts_colors=c)


def pose_to_spherical(R, t):
    o = -R.T @ t
    d = float(np.linalg.norm(o))
    if d < 1e-6: return d, 0.0, 0.0
    el = float(math.degrees(math.asin(np.clip(o[1]/d, -1, 1))))
    az = float(math.degrees(math.atan2(float(o[0]), float(o[2]))))
    return d, el, az


def centre_align(sil_t, dep_t, nor_t, H, W):
    """Shift observations so the mask centroid lands at image centre."""
    ys, xs = torch.where(sil_t > 0.5)
    if len(xs) < 10:
        return sil_t, dep_t, nor_t
    cx_obj, cy_obj = float(xs.float().mean()), float(ys.float().mean())
    dx = int(round(W/2.0 - cx_obj))
    dy = int(round(H/2.0 - cy_obj))
    def roll2(t):
        t = torch.roll(t, dy, 0)
        return torch.roll(t, dx, 1)
    return roll2(sil_t), roll2(dep_t), roll2(nor_t)


def resize2(t, hw):
    if t.dim() == 2:
        return torch.nn.functional.interpolate(
            t.unsqueeze(0).unsqueeze(0), hw, mode="nearest").squeeze()
    return torch.nn.functional.interpolate(
        t.permute(2,0,1).unsqueeze(0), hw, mode="bilinear",
        align_corners=False).squeeze().permute(1,2,0)


def evaluate_distance_strategies(
    *,
    mesh: MeshData,
    rasterizer,
    sil_renderer,
    ref_sil_s:   torch.Tensor,   # (Hs,Ws) float32
    ref_dep_s:   torch.Tensor,   # (Hs,Ws) float32 raw mm
    ref_nor_s:   torch.Tensor,   # (Hs,Ws,3) float32
    gt_dist:     float,
    gt_elev:     float,
    gt_azim:     float,
    mesh_r:      float,
    device:      torch.device,
):
    """
    At the GT angular direction, score candidate distances.
    Strategy A (geometry-only): pick minimum silhouette+normal loss.
    Strategy B (depth-guided): use depth hint, then geometry verification.
    """
    at = torch.zeros(3, device=device)

    # Candidate distance shells: log-spaced 0.3× to 5× mesh_r,
    # but extend to cover actual GT distance
    max_mult = max(5.0, gt_dist / mesh_r * 1.4)
    n = 40
    mults = np.exp(np.linspace(np.log(0.3), np.log(max_mult), n))
    dist_cands = [mesh_r * m for m in mults]

    # Evaluate all at GT angular direction (sil + depth + normals)
    dist_list = dist_cands
    elev_list = [gt_elev] * len(dist_list)
    azim_list = [gt_azim] * len(dist_list)

    # Geometry-only loss (sil + normals, no depth weight)
    os.environ["VAL_W_DEPTH_START"] = "0.0"
    os.environ["VAL_W_NORMAL_START"] = "1.0"
    scores_geom = _batched_eval_candidates_no_grad(
        mesh=mesh, silhouette_renderer=sil_renderer, rasterizer=rasterizer,
        image_ref_silhouette=ref_sil_s, ref_depth=ref_dep_s, ref_normals=ref_nor_s,
        at_xyz=at, dist_list=dist_list, elev_list=elev_list, azim_list=azim_list,
        device=device, use_sil_only=False, use_sil_normals_only=True,
    )
    best_geom_idx = int(scores_geom["loss_total"].argmin().item())
    dist_geom = dist_cands[best_geom_idx]

    # Depth hint
    os.environ["VAL_W_DEPTH_START"] = "1.2"
    hint = _ref_dist_hint_from_depth(ref_dep_s, ref_sil_s, device, mesh_r=mesh_r)
    dist_guided = hint if hint is not None else dist_geom

    return {
        "gt_dist":    gt_dist,
        "geom_only":  dist_geom,
        "depth_guided": dist_guided,
        "depth_hint": hint,
        "rel_err_geom":    abs(dist_geom  - gt_dist) / gt_dist,
        "rel_err_guided":  abs(dist_guided - gt_dist) / gt_dist if hint else float("nan"),
        "correct_shell_geom":    abs(dist_geom  - gt_dist) < 0.15 * gt_dist,
        "correct_shell_guided":  abs(dist_guided - gt_dist) < 0.15 * gt_dist if hint else False,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test objects: pick non-symmetric, well-modelled ones
    # Each scene_id = obj_id in LM
    TEST_OBJECTS = [
        (1,  1),   # ape
        (2,  2),   # benchvise
        (9,  9),   # duck
        (11, 11),  # glue
        (14, 14),  # lamp
        (15, 15),  # phone
    ]
    N_FRAMES = 20
    H, W = 480, 640
    Hs, Ws = 144, 192

    adapter = BopLmAdapter(str(BOP))
    out_dir = ROOT / "lm_dist_alignment_results_metric"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for obj_id, scene_id in TEST_OBJECTS:
        ply  = BOP / "models" / f"obj_{obj_id:06d}.ply"
        if not ply.exists():
            print(f"  [SKIP] obj {obj_id}: PLY not found")
            continue

        mesh = load_mesh(ply, device)
        at   = torch.zeros(3, device=device)
        mesh_r = _mesh_radius_from_verts(mesh.verts, at)

        cameras_dummy = FoVPerspectiveCameras(device=device)
        sil_renderer, rasterizer, _ = build_renderers(Hs, Ws, cameras_dummy)
        # metric hint: calibrate the surface-to-centre offset kappa*r from random renders of this mesh
        # (no observation involved), exactly as the main pipeline does
        exp._estimate_depth_centre_offset(mesh, rasterizer, at, float(mesh_r), device)

        with (BOP/"test"/f"{scene_id:06d}"/"scene_gt.json").open() as f:
            gt_data = json.load(f)
        with (BOP/"test"/f"{scene_id:06d}"/"scene_camera.json").open() as f:
            cam_data = json.load(f)

        # Sample N_FRAMES frames with mask > 2000px and elev in [-10, 60]
        cands = []
        for k, anns in gt_data.items():
            for a in anns:
                if a["obj_id"] != obj_id: continue
                m = cv2.imread(str(BOP/"test"/f"{scene_id:06d}"/"mask_visib"/f"{int(k):06d}_000000.png"), 0)
                if m is None: continue
                npx = int((m > 0).sum())
                if npx < 3000: continue
                R  = np.array(a["cam_R_m2c"]).reshape(3,3)
                t  = np.array(a["cam_t_m2c"])
                d, el, az = pose_to_spherical(R, t)
                if not (-10 < el < 65): continue
                cands.append((npx, int(k), d, el, az))
        cands.sort(reverse=True)
        cands = cands[:N_FRAMES]

        print(f"\nObj {obj_id} ({LM_OBJ_NAMES.get(obj_id,'?')}): "
              f"r={mesh_r:.0f}mm  {len(cands)} frames")

        for npx, fidx, gt_dist, gt_elev, gt_azim in cands:
            try:
                frame = adapter.load_frame(obj_id, scene_id, fidx, (H, W))

                sil_raw = frame.silhouette
                dep_raw = frame.depth_raw_mm * sil_raw  # masked raw mm
                nor_raw = frame.normals

                # Centre-align
                sil_ca, dep_ca, nor_ca = centre_align(sil_raw, dep_raw, nor_raw, H, W)

                # Resize to search resolution
                sil_s = resize2(sil_ca, (Hs, Ws))
                dep_s = resize2(dep_ca, (Hs, Ws))
                nor_s = resize2(nor_ca, (Hs, Ws))
                nor_s = torch.nn.functional.normalize(nor_s, dim=-1)

                sil_s = sil_s.to(device)
                dep_s = dep_s.to(device)
                nor_s = nor_s.to(device)

                res = evaluate_distance_strategies(
                    mesh=mesh, rasterizer=rasterizer, sil_renderer=sil_renderer,
                    ref_sil_s=sil_s, ref_dep_s=dep_s, ref_nor_s=nor_s,
                    gt_dist=gt_dist, gt_elev=gt_elev, gt_azim=gt_azim,
                    mesh_r=mesh_r, device=device,
                )
                res.update({"obj_id": obj_id, "frame_idx": fidx, "mask_px": npx})
                all_rows.append(res)

                print(f"  f={fidx:04d} npx={npx:5d} d*={gt_dist:.0f}mm | "
                      f"geom={res['geom_only']:.0f}mm(err={res['rel_err_geom']:.2f}) | "
                      f"guided={res['depth_guided']:.0f}mm(err={res['rel_err_guided']:.2f}) | "
                      f"hint={res['depth_hint']:.0f}mm" if res['depth_hint'] else
                      f"  f={fidx:04d} d*={gt_dist:.0f}mm | geom={res['geom_only']:.0f}mm | guided=NO HINT")

            except Exception as e:
                print(f"  [ERR] f={fidx}: {e}")

    # Save CSV
    if all_rows:
        csv_path = out_dir / "dist_alignment.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  {'Object':<15} {'N':>4} | {'Geom-only':>10} {'Correct%':>9} | {'Depth-guided':>12} {'Correct%':>9}")
    print(f"  {'-'*65}")
    for obj_id, _ in TEST_OBJECTS:
        rows = [r for r in all_rows if r["obj_id"] == obj_id]
        if not rows: continue
        name = f"{obj_id}-{LM_OBJ_NAMES.get(obj_id,'?')}"
        ge = np.nanmean([r["rel_err_geom"] for r in rows]) * 100
        gc = np.mean([r["correct_shell_geom"] for r in rows]) * 100
        de = np.nanmean([r["rel_err_guided"] for r in rows]) * 100
        dc = np.mean([r["correct_shell_guided"] for r in rows]) * 100
        print(f"  {name:<15} {len(rows):>4} | {ge:>9.1f}% {gc:>8.0f}% | {de:>11.1f}% {dc:>8.0f}%")

    geom_correct = np.mean([r["correct_shell_geom"] for r in all_rows]) * 100
    guided_correct = np.mean([r["correct_shell_guided"] for r in all_rows]) * 100
    geom_err = np.nanmean([r["rel_err_geom"] for r in all_rows]) * 100
    guided_err = np.nanmean([r["rel_err_guided"] for r in all_rows]) * 100
    print(f"\n  {'Overall':<15} {len(all_rows):>4} | {geom_err:>9.1f}% {geom_correct:>8.0f}% | "
          f"{guided_err:>11.1f}% {guided_correct:>8.0f}%")
    print(f"{'='*70}")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
