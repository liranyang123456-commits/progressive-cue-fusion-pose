#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full 6-DoF evaluation of progressive cue fusion on real BOP RGB-D frames
(LM / LM-O / T-LESS), under a documented wide-baseline refinement protocol.

Task
----
Inputs per frame: RGB-D image, intrinsics K, the known metric mesh, the visible mask,
and an initial pose T0 obtained by perturbing the GT pose (rotation offset drawn in a
bin, see --rot-bins). Output: object-to-camera pose (R, t) in BOP convention.
The proposed method does NOT read T0 (it is initialisation-free given the mask); T0 is
exported so that refiner baselines run from exactly the same initialisation.

Method adaptation to a real camera (no learning, no GT)
-------------------------------------------------------
1. Virtual centred camera: the look-at parametrisation assumes the object centre lies on
   the optical axis. We rotate the camera (pure rotation R_v, derived from the observed
   mask centroid only) so that the centroid ray becomes the optical axis, and re-sample
   mask/depth/normals/RGB into a square crop with a pinhole of vertical FOV --fovy.
   Depth is transformed exactly (back-project, rotate, re-project).
2. Roll: the pipeline searches (d, elev, azim) with a fixed up-vector; in-plane roll is
   handled by an observation-rotation bank of K bins (camera roll psi <=> observation
   rotated by -psi), each bin running the same global search; the best bins are refined
   by the same LocalBeam; a final fine roll sweep (1 deg) completes the pose.
3. Conversion: pipeline view frame is OpenGL (x right, y up, z backward); BOP is OpenCV.
   R_cv = F R_view, t_cv = F T with F = diag(1,-1,-1); then undo the virtual rotation.

Metrics (BOP-style, symmetry-aware from models_info.json)
---------------------------------------------------------
rotation geodesic error, translation error, ADD / ADD-S (and recall @0.1 diameter),
MSSD and MSPD recalls averaged over the BOP threshold ranges (AR_MSSD, AR_MSPD).

Usage
-----
  python bop_full6dof_eval.py --selftest
  python bop_full6dof_eval.py --dataset lm  --objects 1,5,6,8,9,10,11,12  --frames-per-object 4 --out bop6dof_results/lm
  python bop_full6dof_eval.py --dataset tless --objects 1,5,9,14,20,23,27,30 --frames-per-object 4 --out bop6dof_results/tless
  python bop_full6dof_eval.py --dataset lm --export-only --out bop6dof_results/lm   # package inputs for baselines
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["VAL_DISABLE_TORCH_COMPILE"] = "1"
os.environ["FORCE_CODE_RUN_PARAMS"] = "0"
os.environ.setdefault("VAL_STRICT_NO_GT", "1")
os.environ.setdefault("VAL_DEBUG_RAST", "0")

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    import Exp_Camera_Optim_RT_Val_Beam_Search12291_Standford122 as exp  # noqa: E402

F_GL2CV = np.diag([1.0, -1.0, -1.0])

# --------------------------------------------------------------------------------------
# Dataset access
# --------------------------------------------------------------------------------------
BOP_ROOT = Path(os.environ.get("BOP_ROOT", os.environ.get("BOP_ROOT", "data/bop")))
DATASETS = {
    "lm": {"root": BOP_ROOT / "lm", "test": "test", "models": "models", "models_eval": "models_eval", "depth_scale": 1.0},
    "lmo": {"root": BOP_ROOT / "lmo", "test": "test", "models": "models", "models_eval": "models_eval", "depth_scale": 1.0},
    "tless": {"root": BOP_ROOT / "tless", "test": "test_primesense", "models": "models_cad", "models_eval": "models_eval", "depth_scale": 0.1},
}


@dataclass
class Frame:
    dataset: str
    scene_id: int
    im_id: int
    obj_id: int
    inst_idx: int
    K: np.ndarray
    depth_scale: float
    R_gt: np.ndarray
    t_gt: np.ndarray  # mm
    rgb: np.ndarray  # (H,W,3) float [0,1]
    depth: np.ndarray  # (H,W) mm, 0 invalid
    mask: np.ndarray  # (H,W) bool visible mask


def _load_json(p: Path) -> dict:
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def list_test_targets(ds: str) -> list[tuple[int, int, int]]:
    """(scene_id, im_id, obj_id) from test_targets_bop19.json if present, else from scene_gt."""
    cfg = DATASETS[ds]
    root = cfg["root"]
    out: list[tuple[int, int, int]] = []
    tt = root / "test_targets_bop19.json"
    if tt.exists():
        for e in _load_json(tt):
            out.append((int(e["scene_id"]), int(e["im_id"]), int(e["obj_id"])))
        return out
    for scene_dir in sorted((root / cfg["test"]).glob("*")):
        if not scene_dir.is_dir():
            continue
        gt = _load_json(scene_dir / "scene_gt.json")
        for im_id, anns in gt.items():
            for a in anns:
                out.append((int(scene_dir.name), int(im_id), int(a["obj_id"])))
    return out


def load_frame(ds: str, scene_id: int, im_id: int, obj_id: int) -> Frame:
    cfg = DATASETS[ds]
    scene_dir = cfg["root"] / cfg["test"] / f"{scene_id:06d}"
    cam = _load_json(scene_dir / "scene_camera.json")[str(im_id)]
    K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
    depth_scale = float(cam.get("depth_scale", cfg["depth_scale"]))
    anns = _load_json(scene_dir / "scene_gt.json")[str(im_id)]
    inst_idx = next(i for i, a in enumerate(anns) if int(a["obj_id"]) == obj_id)
    a = anns[inst_idx]
    R_gt = np.array(a["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    t_gt = np.array(a["cam_t_m2c"], dtype=np.float64).reshape(3)
    rgb_p = scene_dir / "rgb" / f"{im_id:06d}.png"
    if not rgb_p.exists():
        rgb_p = scene_dir / "rgb" / f"{im_id:06d}.jpg"
    rgb = cv2.cvtColor(cv2.imread(str(rgb_p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = cv2.imread(str(scene_dir / "depth" / f"{im_id:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) * depth_scale
    mask = cv2.imread(str(scene_dir / "mask_visib" / f"{im_id:06d}_{inst_idx:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
    return Frame(ds, scene_id, im_id, obj_id, inst_idx, K, depth_scale, R_gt, t_gt, rgb, depth, mask)


def load_frame_from_export(fdir: Path) -> Frame:
    """Rebuild the Frame from observation.npz + protocol.json written by --export-only (identical content)."""
    proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
    obs = np.load(fdir / "observation.npz")
    rgb = obs["rgb"].astype(np.float32) / 255.0
    return Frame(proto["dataset"], int(proto["scene_id"]), int(proto["im_id"]), int(proto["obj_id"]), int(proto.get("inst_idx", 0)),
                 np.asarray(obs["K"], dtype=np.float64), 1.0, np.array(proto["gt_R"], dtype=np.float64), np.array(proto["gt_t"], dtype=np.float64),
                 rgb, obs["depth"].astype(np.float32), obs["mask"].astype(bool))


def select_frames_from_export(out_root: Path) -> list[tuple[int, int, int]]:
    out = []
    for d in sorted(out_root.iterdir()):
        if d.is_dir() and (d / "protocol.json").exists() and (d / "observation.npz").exists():
            pr = json.loads((d / "protocol.json").read_text(encoding="utf-8"))
            out.append((int(pr["scene_id"]), int(pr["im_id"]), int(pr["obj_id"])))
    return out


def load_model(ds: str, obj_id: int, device: torch.device):
    import open3d as o3d

    cfg = DATASETS[ds]
    ply = cfg["root"] / cfg["models"] / f"obj_{obj_id:06d}.ply"
    m = o3d.io.read_triangle_mesh(str(ply))
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    tris = np.asarray(m.triangles)
    if tris.shape[0] > 80000:
        m = m.simplify_quadric_decimation(80000)
    m.compute_vertex_normals()
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.triangles, dtype=np.int64)
    vn = np.asarray(m.vertex_normals, dtype=np.float32)
    colors = np.asarray(m.vertex_colors, dtype=np.float32) if m.has_vertex_colors() else None
    info = _load_json(cfg["root"] / cfg["models"] / "models_info.json")[str(obj_id)]
    mesh = exp.MeshData(
        verts=torch.from_numpy(verts).to(device),
        faces=torch.from_numpy(faces).to(device),
        verts_normals=torch.from_numpy(vn).to(device),
        verts_colors=torch.from_numpy(colors).to(device) if colors is not None else None,
    )
    # evaluation point set (BOP models_eval if available, else sub-sampled vertices)
    eval_ply = cfg["root"] / cfg["models_eval"] / f"obj_{obj_id:06d}.ply"
    pts = np.asarray(o3d.io.read_triangle_mesh(str(eval_ply)).vertices, dtype=np.float64) if eval_ply.exists() else verts.astype(np.float64)
    if pts.shape[0] > 3000:
        pts = pts[np.random.default_rng(0).choice(pts.shape[0], 3000, replace=False)]
    return mesh, info, pts


# --------------------------------------------------------------------------------------
# Symmetry-aware metrics: shared with every baseline runner (bop_metrics.py)
# --------------------------------------------------------------------------------------
from bop_metrics import symmetry_transforms, transform_pts, project_pts, compute_metrics  # noqa: E402


# --------------------------------------------------------------------------------------
# Wide-baseline initial pose (protocol) -- exported for refiner baselines
# --------------------------------------------------------------------------------------
def perturb_pose(R_gt, t_gt, rot_bin: tuple[float, float], rng: np.random.Generator,
                 dist_scale: tuple[float, float] = (0.7, 1.4), lateral_frac: float = 0.05):
    ang = math.radians(rng.uniform(*rot_bin))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    dR = cv2.Rodrigues(axis * ang)[0]
    R0 = dR @ R_gt
    d = np.linalg.norm(t_gt)
    s = rng.uniform(*dist_scale)
    lat = rng.normal(size=3)
    lat -= lat.dot(t_gt / d) * (t_gt / d)
    lat = lat / (np.linalg.norm(lat) + 1e-9) * rng.uniform(0, lateral_frac) * d
    t0 = t_gt * s + lat
    rot_err = math.degrees(math.acos(np.clip((np.trace(R0 @ R_gt.T) - 1) / 2, -1, 1)))
    return R0, t0, rot_err


# --------------------------------------------------------------------------------------
# Virtual centred camera and observation warping (observation-only; no GT)
# --------------------------------------------------------------------------------------
def rotation_to_axis(v: np.ndarray, target=np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Minimal rotation R with R v = target."""
    v = v / np.linalg.norm(v)
    c = float(np.clip(v.dot(target), -1, 1))
    ax = np.cross(v, target)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return np.eye(3) if c > 0 else cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    return cv2.Rodrigues(ax / s * math.atan2(s, c))[0]


def normals_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    d = cv2.GaussianBlur(depth, (5, 5), 1.0)
    d = np.where(depth > 0, d, 0.0)
    xs = (np.arange(W) - K[0, 2]) / K[0, 0]
    ys = (np.arange(H) - K[1, 2]) / K[1, 1]
    xg, yg = np.meshgrid(xs, ys)
    P = np.stack([xg * d, yg * d, d], -1)
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, :-1] = P[:, 1:] - P[:, :-1]
    dy[:-1] = P[1:] - P[:-1]
    n = np.cross(dy, dx)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    valid = (nn[..., 0] > 1e-6) & (d > 0)
    n = np.where(valid[..., None], n / np.maximum(nn, 1e-9), 0.0)
    # face the camera (camera looks along +z in OpenCV): n.z < 0 for visible surfaces
    flip = n[..., 2] > 0
    n[flip] *= -1
    return n.astype(np.float32)


@dataclass
class VirtualObs:
    R_v: np.ndarray  # original camera -> virtual camera rotation
    K_v: np.ndarray
    size: int
    fovy_deg: float
    mask: np.ndarray  # (S,S) float 0/1  (OpenCV image convention, row 0 = top)
    depth: np.ndarray  # (S,S) mm metric view depth in the virtual camera
    normals: np.ndarray  # (S,S,3) in virtual OpenCV camera frame
    rgb: np.ndarray  # (S,S,3)


def make_virtual_obs(fr: Frame, size: int = 192, fovy_deg: float = 40.0, pad: float = 1.6) -> VirtualObs:
    ys, xs = np.where(fr.mask)
    if len(xs) < 20:
        raise ValueError("mask too small")
    uc, vc = float(xs.mean()), float(ys.mean())
    K = fr.K
    ray = np.linalg.inv(K) @ np.array([uc, vc, 1.0])
    R_v = rotation_to_axis(ray)
    # virtual FOV: fit the mask extent with padding, but not below --fovy to keep the search scale similar
    ext = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    fov_fit = 2 * math.degrees(math.atan(pad * 0.5 * ext / float(K[1, 1])))
    fovy = max(min(fov_fit, 90.0), 10.0)
    f_v = (size / 2.0) / math.tan(math.radians(fovy) / 2.0)
    K_v = np.array([[f_v, 0, size / 2.0], [0, f_v, size / 2.0], [0, 0, 1.0]])

    # inverse warp: for every virtual pixel find the source pixel
    uu, vv = np.meshgrid(np.arange(size), np.arange(size))
    rays_v = np.stack([(uu - K_v[0, 2]) / K_v[0, 0], (vv - K_v[1, 2]) / K_v[1, 1], np.ones_like(uu, dtype=np.float64)], -1)
    rays_c = rays_v @ R_v  # R_v^T applied to each ray (row-vector form)
    z = rays_c[..., 2]
    us = K[0, 0] * rays_c[..., 0] / np.where(z > 1e-9, z, 1e-9) + K[0, 2]
    vs = K[1, 1] * rays_c[..., 1] / np.where(z > 1e-9, z, 1e-9) + K[1, 2]
    map_x = us.astype(np.float32)
    map_y = vs.astype(np.float32)
    mask_v = cv2.remap(fr.mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderValue=0) > 0
    depth_src = cv2.remap(fr.depth, map_x, map_y, cv2.INTER_NEAREST, borderValue=0)
    rgb_v = cv2.remap(fr.rgb, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    # exact depth transform: point in original camera along the sampled source ray, rotate
    src_ray = np.stack([(map_x - K[0, 2]) / K[0, 0], (map_y - K[1, 2]) / K[1, 1], np.ones_like(map_x)], -1).astype(np.float64)
    P_c = src_ray * depth_src[..., None]
    P_v = P_c @ R_v.T
    depth_v = np.where((depth_src > 0) & mask_v, P_v[..., 2], 0.0).astype(np.float32)
    normals_v = normals_from_depth(depth_v, K_v) * mask_v[..., None]
    return VirtualObs(R_v=R_v, K_v=K_v, size=size, fovy_deg=fovy, mask=mask_v.astype(np.float32),
                      depth=depth_v, normals=normals_v, rgb=rgb_v.astype(np.float32))


def rotate_obs(obs: VirtualObs, psi_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Camera roll psi (R_z(psi) in the OpenCV virtual frame) <=> rotate the observation by -psi
    (cv2 angle convention, counter-clockwise on screen). Returns (mask, depth, normals, rgb)."""
    S = obs.size
    M = cv2.getRotationMatrix2D((S / 2.0 - 0.5, S / 2.0 - 0.5), -psi_deg, 1.0)
    mask = cv2.warpAffine(obs.mask, M, (S, S), flags=cv2.INTER_NEAREST, borderValue=0)
    depth = cv2.warpAffine(obs.depth, M, (S, S), flags=cv2.INTER_NEAREST, borderValue=0)
    rgb = cv2.warpAffine(obs.rgb, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
    n = cv2.warpAffine(obs.normals, M, (S, S), flags=cv2.INTER_NEAREST, borderValue=0)
    c, s = math.cos(math.radians(psi_deg)), math.sin(math.radians(psi_deg))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    n = n @ Rz.T
    n = n * (mask > 0.5)[..., None]
    return mask, depth, n, rgb


def obs_to_pipeline(mask, depth, normals, rgb, device):
    """OpenCV image (row 0 top, normals in OpenCV cam frame) -> pipeline tensors
    (nvdiffrast image rows bottom-up; view frame x right, y up, z backward)."""
    m = torch.from_numpy(np.ascontiguousarray(mask[::-1])).float().to(device)
    d = torch.from_numpy(np.ascontiguousarray(depth[::-1])).float().to(device)
    n = normals[::-1].copy()
    n = n @ F_GL2CV.astype(np.float32)  # n_gl = F n_cv (F symmetric)
    n = torch.from_numpy(np.ascontiguousarray(n)).float().to(device)
    r = torch.from_numpy(np.ascontiguousarray(rgb[::-1])).float().to(device)
    return m, d, n, r


def pipeline_pose_to_cv(dist, elev, azim, psi_deg, R_v, device) -> tuple[np.ndarray, np.ndarray]:
    """(d, elev, azim) look-at pose in the pipeline frame + roll psi in the virtual OpenCV camera
    -> object-to-camera (R, t) in the ORIGINAL BOP camera."""
    at = torch.zeros(3, device=device)
    R, T = exp.look_at_view_transform(dist=float(dist), elev=float(elev), azim=float(azim), at=at, device=device)
    R_view = R[0].detach().cpu().numpy().astype(np.float64)
    T_view = T[0].detach().cpu().numpy().astype(np.float64)
    R_cv = F_GL2CV @ R_view
    t_cv = F_GL2CV @ T_view
    # The pipeline saw the observation of the ROLLED camera X' = R_z(psi) X_v (see rotate_obs), so its
    # look-at pose is the pose of that rolled camera; the virtual camera is X_v = R_z(-psi) X'.
    c, s = math.cos(math.radians(-psi_deg)), math.sin(math.radians(-psi_deg))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R_virt = Rz @ R_cv
    t_virt = Rz @ t_cv
    # virtual camera -> original camera: X_c = R_v^T X_v
    R_orig = R_v.T @ R_virt
    t_orig = R_v.T @ t_virt
    return R_orig, t_orig


def splat_render(verts: np.ndarray, faces: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, H: int, W: int):
    """Independent (nvdiffrast-free) z-buffer point-splat renderer in OpenCV convention.
    Returns (mask bool (H,W), depth (H,W)). Used only for self-tests / sanity checks."""
    tri = verts[faces]  # (F,3,3)
    cen = tri.mean(axis=1)
    pts = np.concatenate([verts, cen, (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2, (tri[:, 0] + tri[:, 2]) / 2], 0)
    pc = pts @ R.T + t
    z = pc[:, 2]
    ok = z > 1e-6
    uv = project_pts(K, pc[ok])
    zz = z[ok]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    depth = np.full((H, W), np.inf)
    np.minimum.at(depth, (v[inb], u[inb]), zz[inb])
    mask = np.isfinite(depth)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    depth = np.where(np.isfinite(depth), depth, 0.0)
    return mask, depth


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0.5
    b = b > 0.5
    return float((a & b).sum() / max((a | b).sum(), 1))


# --------------------------------------------------------------------------------------
# The method on one virtual observation (roll bank + global + beam + roll sweep)
# --------------------------------------------------------------------------------------
PIPE_ENV = {
    "VAL_RUN_PROFILE": "v10",
    "VAL_DETERMINISTIC": "1",
    "VAL_RANDOM_SEED": "12291",
    "VAL_STRICT_NO_GT": "1",
    "VAL_RAND_RADII_MULTS": "auto",
    "VAL_RAND_SAMPLES_PER_RADIUS": "512",
    "VAL_RAND_RADIUS_STEPS": "24",
    "VAL_RAND_BATCH": "256",
    "VAL_RAND_TOPK": "16",
    "VAL_TOPK": "16",
    "VAL_LOSS_SCALE": "auto",
    "VAL_LOCAL_BEAM_DIST_SCALES": "auto",
    "VAL_REF_DIST_PENALTY_RELATIVE": "1",
    "VAL_REF_DIST_PENALTY_WEIGHT": "0.55",
    "VAL_LOCAL_BEAM_WIDTH": "48",
    "VAL_LOCAL_BEAM_LEVELS": "32",
    "VAL_LOCAL_BEAM_EARLY_STOP": "1",
    "VAL_LOCAL_BEAM_EARLY_STOP_PATIENCE": "8",
    "VAL_LOCAL_BEAM_SCORE_MODE": "full",
    "VAL_LOCAL_BEAM_FINAL_DR_ENABLE": "0",
    "VAL_SELECT_USE_VISUAL_SCORE": "0",  # appearance ranking not used on real RGB (domain gap)
    "VAL_RAND_VERBOSE": "0",
    "VAL_LOCAL_BEAM_VERBOSE": "0",
    "VAL_GLOBAL_VERBOSE": "0",
    "VAL_W_SIL_START": "1.0",
    "VAL_W_DEPTH_START": "1.5",
    "VAL_W_NORMAL_START": "0.8",
    "VAL_W_IOU": "0.4",
    "VAL_GT_ELEV_MIN": "-89.0",  # real cameras may look from below the object frame's equator
    "VAL_GT_ELEV_MAX": "89.0",
    "VAL_LOCAL_BEAM_ELEV_CONSTRAIN": "0",
    "VAL_LOCAL_BEAM_AZIM_CONSTRAIN": "0",
}


def run_method(mesh, mesh_r: float, obs: VirtualObs, device, out_dir: Path, *, roll_bins: int = 12,
               beam_seeds: int = 8, verbose: bool = True) -> dict:
    """Returns dict with est pose in the virtual camera (dist, elev, azim, psi) and timings."""
    for k, v in PIPE_ENV.items():
        os.environ[k] = v
    os.environ["VAL_FOVY_DEG"] = f"{obs.fovy_deg:.6f}"
    S = obs.size
    at = torch.zeros(3, device=device)
    cams = exp.FoVPerspectiveCameras(device=device)
    sil_r, rast, phong = exp.build_renderers(S, S, cams, faces_per_pixel_silhouette=1)
    exp._estimate_depth_centre_offset(mesh, rast, at, float(mesh_r), device)
    t0 = exp._stage_now()

    # Stage I over the roll bank
    cand = []  # (score, dist, elev, azim, psi)
    for k in range(roll_bins):
        psi = 360.0 * k / roll_bins
        m, d, n, r = rotate_obs(obs, psi)
        mt, dt, nt, rt = obs_to_pipeline(m, d, n, r, device)
        seeds = exp.global_random_sampling_1024(
            mesh=mesh, verts=mesh.verts, mesh_center_xyz=at, silhouette_renderer=sil_r, rasterizer=rast,
            phong_renderer=None, image_ref_silhouette=mt, ref_depth=dt, ref_normals=nt, image_ref_rgb=None,
            init_dist=float(3.0 * mesh_r), init_elev=20.0, init_azim=0.0, device=device,
            save_dir=str(out_dir / f"roll_{k:02d}"), tag=f"roll{k:02d}", R_gt=None, T_gt=None,
        )
        for (sd, se, sa, sl) in seeds:
            if math.isfinite(sl):
                cand.append((float(sl), float(sd), float(se), float(sa), psi))
    t1 = exp._stage_now()
    cand.sort(key=lambda x: x[0])
    # keep the best seeds overall but at least one per distinct roll among the top bins
    chosen = []
    seen_roll = set()
    for c in cand:
        if len(chosen) >= beam_seeds:
            break
        if c[4] in seen_roll and sum(1 for x in chosen if x[4] == c[4]) >= 2:
            continue
        chosen.append(c)
        seen_roll.add(c[4])
    if verbose:
        print(f"  [StageI] {len(cand)} seeds over {roll_bins} roll bins; chosen rolls={sorted(set(c[4] for c in chosen))}  {t1-t0:.1f}s")

    # Stage II+III per seed (distance sweep inside LocalBeam, coarse + fine), observation per roll
    results = []
    obs_cache: dict[float, tuple] = {}
    for i, (sl, sd, se, sa, psi) in enumerate(chosen):
        if psi not in obs_cache:
            m, d, n, r = rotate_obs(obs, psi)
            obs_cache[psi] = obs_to_pipeline(m, d, n, r, device)
        mt, dt, nt, rt = obs_cache[psi]
        run_dir = out_dir / f"seed_{i:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        prev = {k: os.environ.get(k, "") for k in ("VAL_LOCAL_BEAM_STEP_FRAC", "VAL_LOCAL_BEAM_LEVELS", "VAL_LOCAL_BEAM_WIDTH")}
        os.environ["VAL_LOCAL_BEAM_STEP_FRAC"], os.environ["VAL_LOCAL_BEAM_LEVELS"], os.environ["VAL_LOCAL_BEAM_WIDTH"] = "0.08", "8", "24"
        c_d, c_e, c_a, c_l = exp.local_beam_search_on_sphere(
            mesh=mesh, silhouette_renderer=sil_r, rasterizer=rast, phong_renderer=None,
            image_ref_silhouette=mt, ref_depth=dt, ref_normals=nt, image_ref_rgb=None, at_xyz=at,
            start_dist=sd, start_elev=se, start_azim=sa, device=device, save_dir=str(run_dir), tag=f"s{i:02d}_coarse",
        )
        for k, v in prev.items():
            if v != "":
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        sd2, se2, sa2 = (c_d, c_e, c_a) if c_l < sl else (sd, se, sa)
        f_d, f_e, f_a, f_l = exp.local_beam_search_on_sphere(
            mesh=mesh, silhouette_renderer=sil_r, rasterizer=rast, phong_renderer=None,
            image_ref_silhouette=mt, ref_depth=dt, ref_normals=nt, image_ref_rgb=None, at_xyz=at,
            start_dist=sd2, start_elev=se2, start_azim=sa2, device=device, save_dir=str(run_dir), tag=f"s{i:02d}_fine",
        )
        results.append((float(f_l), float(f_d), float(f_e), float(f_a), float(psi)))
    t2 = exp._stage_now()
    results.sort(key=lambda x: x[0])
    best_l, best_d, best_e, best_a, best_psi = results[0]

    # Final 1-degree roll sweep around the winning bin (geometry loss on the rotated observation)
    half = 180.0 / roll_bins
    best = (best_l, best_psi)
    for dpsi in np.arange(-half, half + 1e-6, 1.0):
        psi = best_psi + float(dpsi)
        m, d, n, r = rotate_obs(obs, psi)
        mt, dt, nt, rt = obs_to_pipeline(m, d, n, r, device)
        ev = exp._evaluate_pose_loss_no_grad(mesh, sil_r, rast, mt, dt, nt, at, best_d, best_e, best_a, device)
        l = float(ev["loss_total"])
        if l < best[0]:
            best = (l, psi)
    best_l, best_psi = best

    # Joint coordinate-descent polish over (elev, azim, roll) with the geometry loss
    # (distance fixed): steps 3 -> 1.5 -> 0.75 deg, accept only strict improvements.
    obs_cache2: dict[float, tuple] = {}

    def _eval(e: float, a: float, psi: float) -> float:
        key = round(psi, 3)
        if key not in obs_cache2:
            m, d, n, r = rotate_obs(obs, psi)
            obs_cache2[key] = obs_to_pipeline(m, d, n, r, device)
        mt, dt, nt, rt = obs_cache2[key]
        ev = exp._evaluate_pose_loss_no_grad(mesh, sil_r, rast, mt, dt, nt, at, best_d, e, a, device)
        return float(ev["loss_total"])

    cur = [best_e, best_a, best_psi]
    cur_l = _eval(*cur)
    for step in (3.0, 1.5, 0.75):
        for _ in range(4):
            improved = False
            for k in range(3):
                for sgn in (+1.0, -1.0):
                    cand = list(cur)
                    cand[k] += sgn * step
                    if k == 0:
                        cand[0] = max(-89.0, min(89.0, cand[0]))
                    l = _eval(*cand)
                    if l < cur_l - 1e-9:
                        cur, cur_l, improved = cand, l, True
            if not improved:
                break
    best_e, best_a, best_psi = cur
    best_l = cur_l
    t3 = exp._stage_now()
    return {
        "dist": best_d, "elev": best_e, "azim": best_a, "psi": best_psi, "loss": best_l,
        "t_global_s": t1 - t0, "t_local_s": t2 - t1, "t_roll_s": t3 - t2, "t_total_s": t3 - t0,
        "n_seeds": len(chosen), "roll_bins": roll_bins,
    }


def cv_pose_to_pipeline(R_orig: np.ndarray, t_orig: np.ndarray, R_v: np.ndarray, device) -> tuple[float, float, float, float]:
    """Inverse of pipeline_pose_to_cv: object-to-camera pose in the ORIGINAL BOP camera ->
    (dist, elev, azim) of the zero-roll look-at camera in the pipeline frame + roll psi (deg)."""
    R_virt = R_v @ R_orig
    t_virt = R_v @ t_orig
    c_obj = -R_virt.T @ t_virt  # camera centre in the (centred) object frame, OpenCV == object frame
    d = float(np.linalg.norm(c_obj))
    elev = math.degrees(math.asin(float(np.clip(c_obj[1] / max(d, 1e-9), -1, 1))))
    azim = math.degrees(math.atan2(float(c_obj[0]), float(c_obj[2])))
    at = torch.zeros(3, device=device)
    R, T = exp.look_at_view_transform(dist=d, elev=elev, azim=azim, at=at, device=device)
    R_cv0 = F_GL2CV @ R[0].detach().cpu().numpy().astype(np.float64)
    # R_virt ~= Rz(-psi) R_cv0  ->  Rz(-psi) = R_virt R_cv0^T ; angle of that z-rotation
    Rz = R_virt @ R_cv0.T
    minus_psi = math.degrees(math.atan2(Rz[1, 0], Rz[0, 0]))
    return d, elev, azim, -minus_psi


def run_from_init(mesh, mesh_r: float, obs: VirtualObs, device, out_dir: Path, R0: np.ndarray, t0: np.ndarray, *,
                  roll_bins: int = 12, verbose: bool = True) -> dict:
    """Stages II-IV only (distance-aware LocalBeam coarse+fine, roll sweep, joint polish) from a given
    object-to-camera initial pose in the original BOP camera. Used for (i) the refiner-style protocol
    (same T0 bins as WAPR-refine / Diff-DOPE) and (ii) foundation-model initialisations."""
    for k, v in PIPE_ENV.items():
        os.environ[k] = v
    os.environ["VAL_FOVY_DEG"] = f"{obs.fovy_deg:.6f}"
    S = obs.size
    at = torch.zeros(3, device=device)
    cams = exp.FoVPerspectiveCameras(device=device)
    sil_r, rast, phong = exp.build_renderers(S, S, cams, faces_per_pixel_silhouette=1)
    exp._estimate_depth_centre_offset(mesh, rast, at, float(mesh_r), device)
    t0s = exp._stage_now()
    sd, se, sa, psi = cv_pose_to_pipeline(R0, t0, obs.R_v, device)
    se = max(-89.0, min(89.0, se))
    m, d, n, r = rotate_obs(obs, psi)
    mt, dt, nt, rt = obs_to_pipeline(m, d, n, r, device)
    run_dir = out_dir / "init_refine"
    run_dir.mkdir(parents=True, exist_ok=True)
    l0 = float(exp._evaluate_pose_loss_no_grad(mesh, sil_r, rast, mt, dt, nt, at, sd, se, sa, device)["loss_total"])
    prev = {k: os.environ.get(k, "") for k in ("VAL_LOCAL_BEAM_STEP_FRAC", "VAL_LOCAL_BEAM_LEVELS", "VAL_LOCAL_BEAM_WIDTH")}
    os.environ["VAL_LOCAL_BEAM_STEP_FRAC"], os.environ["VAL_LOCAL_BEAM_LEVELS"], os.environ["VAL_LOCAL_BEAM_WIDTH"] = "0.08", "8", "24"
    c_d, c_e, c_a, c_l = exp.local_beam_search_on_sphere(
        mesh=mesh, silhouette_renderer=sil_r, rasterizer=rast, phong_renderer=None,
        image_ref_silhouette=mt, ref_depth=dt, ref_normals=nt, image_ref_rgb=None, at_xyz=at,
        start_dist=sd, start_elev=se, start_azim=sa, device=device, save_dir=str(run_dir), tag="init_coarse",
    )
    for k, v in prev.items():
        if v != "":
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    sd2, se2, sa2 = (c_d, c_e, c_a) if c_l < l0 else (sd, se, sa)
    f_d, f_e, f_a, f_l = exp.local_beam_search_on_sphere(
        mesh=mesh, silhouette_renderer=sil_r, rasterizer=rast, phong_renderer=None,
        image_ref_silhouette=mt, ref_depth=dt, ref_normals=nt, image_ref_rgb=None, at_xyz=at,
        start_dist=sd2, start_elev=se2, start_azim=sa2, device=device, save_dir=str(run_dir), tag="init_fine",
    )
    best_l, best_d, best_e, best_a, best_psi = float(f_l), float(f_d), float(f_e), float(f_a), float(psi)
    t1s = exp._stage_now()
    half = 180.0 / roll_bins
    best = (best_l, best_psi)
    for dpsi in np.arange(-half, half + 1e-6, 1.0):
        ps = best_psi + float(dpsi)
        m, d, n, r = rotate_obs(obs, ps)
        mt, dt, nt, rt = obs_to_pipeline(m, d, n, r, device)
        l = float(exp._evaluate_pose_loss_no_grad(mesh, sil_r, rast, mt, dt, nt, at, best_d, best_e, best_a, device)["loss_total"])
        if l < best[0]:
            best = (l, ps)
    best_l, best_psi = best
    obs_cache2: dict[float, tuple] = {}

    def _eval(e: float, a: float, ps: float) -> float:
        key = round(ps, 3)
        if key not in obs_cache2:
            m, d, n, r = rotate_obs(obs, ps)
            obs_cache2[key] = obs_to_pipeline(m, d, n, r, device)
        mt, dt, nt, rt = obs_cache2[key]
        return float(exp._evaluate_pose_loss_no_grad(mesh, sil_r, rast, mt, dt, nt, at, best_d, e, a, device)["loss_total"])

    cur = [best_e, best_a, best_psi]
    cur_l = _eval(*cur)
    for step in (3.0, 1.5, 0.75):
        for _ in range(4):
            improved = False
            for k in range(3):
                for sgn in (+1.0, -1.0):
                    cand = list(cur)
                    cand[k] += sgn * step
                    if k == 0:
                        cand[0] = max(-89.0, min(89.0, cand[0]))
                    l = _eval(*cand)
                    if l < cur_l - 1e-9:
                        cur, cur_l, improved = cand, l, True
            if not improved:
                break
    best_e, best_a, best_psi = cur
    t2s = exp._stage_now()
    if verbose:
        print(f"  [init-refine] start (d={sd:.0f}, e={se:.1f}, a={sa:.1f}, psi={psi:.1f}) loss {l0:.3f} -> {cur_l:.3f}  {t2s - t0s:.1f}s")
    return {"dist": best_d, "elev": best_e, "azim": best_a, "psi": best_psi, "loss": float(cur_l), "loss_init": l0,
            "t_global_s": 0.0, "t_local_s": t1s - t0s, "t_roll_s": t2s - t1s, "t_total_s": t2s - t0s, "n_seeds": 1, "roll_bins": roll_bins}


# --------------------------------------------------------------------------------------
# Self-test: conventions + roll handling on a synthetic observation (independent renderer)
# --------------------------------------------------------------------------------------
def selftest(device) -> int:
    ds, obj = "lm", 1
    mesh, info, pts = load_model(ds, obj, device)
    verts = mesh.verts.cpu().numpy().astype(np.float64)
    faces = mesh.faces.cpu().numpy()
    centre = verts.mean(axis=0)
    mesh_c = exp.MeshData(verts=mesh.verts - torch.from_numpy(centre).float().to(device), faces=mesh.faces,
                          verts_normals=mesh.verts_normals, verts_colors=mesh.verts_colors)
    verts_c = verts - centre
    r = float(np.max(np.linalg.norm(verts_c, axis=1)))
    # (1) convention check: pipeline look-at -> OpenCV pose; render with the independent splat renderer
    S, fovy = 192, 40.0
    f_v = (S / 2) / math.tan(math.radians(fovy) / 2)
    K_v = np.array([[f_v, 0, S / 2], [0, f_v, S / 2], [0, 0, 1.0]])
    os.environ["VAL_FOVY_DEG"] = str(fovy)
    os.environ["VAL_DEBUG_RAST"] = "0"
    cams = exp.FoVPerspectiveCameras(device=device)
    sil_r, rast, _ = exp.build_renderers(S, S, cams, faces_per_pixel_silhouette=1)
    at = torch.zeros(3, device=device)
    d0, e0, a0 = 6.0 * r, 25.0, 40.0
    R, T = exp.look_at_view_transform(dist=d0, elev=e0, azim=a0, at=at, device=device)
    with torch.no_grad():
        frg = rast(meshes_world=mesh_c, R=R, T=T)
        m_pipe = (frg.pix_to_face[..., 0][0] >= 0).float().cpu().numpy()[::-1]  # to row-0-top
        d_pipe = (frg.zbuf[..., 0][0]).cpu().numpy()[::-1]
    R_cv, t_cv = pipeline_pose_to_cv(d0, e0, a0, 0.0, np.eye(3), device)
    m_spl, d_spl = splat_render(verts_c, faces, K_v, R_cv, t_cv, S, S)
    iou0 = iou(m_pipe, m_spl)
    both = (m_pipe > 0.5) & m_spl & (d_spl > 0)
    dep_err = float(np.median(np.abs(d_pipe[both] - d_spl[both]))) if both.any() else float("nan")
    print(f"[selftest] convention IoU(pipeline vs OpenCV splat)={iou0:.3f}  median|depth diff|={dep_err:.3f} mm (r={r:.1f})")
    ok = iou0 > 0.85 and dep_err < 0.02 * r

    # (2) roll handling: build an OpenCV-convention observation at a GT pose with roll, run the method
    psi_true = 37.0
    c, s = math.cos(math.radians(psi_true)), math.sin(math.radians(psi_true))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R_gt, t_gt = Rz @ R_cv, Rz @ t_cv
    m_gt, d_gt = splat_render(verts_c, faces, K_v, R_gt, t_gt, S, S)
    # splat depth is sparse-ish; fill holes by closing on depth
    d_fill = cv2.morphologyEx(d_gt.astype(np.float32), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    d_fill = np.where(m_gt, np.where(d_gt > 0, d_gt, d_fill), 0.0).astype(np.float32)
    obs = VirtualObs(R_v=np.eye(3), K_v=K_v, size=S, fovy_deg=fovy, mask=m_gt.astype(np.float32), depth=d_fill,
                     normals=normals_from_depth(d_fill, K_v) * m_gt[..., None], rgb=np.zeros((S, S, 3), np.float32))
    out = ROOT / "bop6dof_results" / "_selftest"
    out.mkdir(parents=True, exist_ok=True)
    res = run_method(mesh_c, r, obs, device, out, roll_bins=12, beam_seeds=6)
    R_est, t_est = pipeline_pose_to_cv(res["dist"], res["elev"], res["azim"], res["psi"], np.eye(3), device)
    met = compute_metrics(R_est, t_est, R_gt, t_gt, K_v, pts - centre, info, symmetry_transforms(info))
    print(f"[selftest] roll psi_true={psi_true} est={res['psi']:.1f}; e_R={met['e_R_deg']:.2f} deg, e_t={met['e_t_mm']:.2f} mm, ADD={met['add']:.2f} mm ({res['t_total_s']:.0f}s)")
    ok = ok and met["e_R_deg"] < 10.0
    print("[selftest]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
# Main evaluation loop
# --------------------------------------------------------------------------------------
def select_frames(ds: str, objects: list[int], per_object: int, seed: int) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    targets = list_test_targets(ds)
    out = []
    for obj in objects:
        cands = sorted({(s, i, o) for (s, i, o) in targets if o == obj})
        if not cands:
            continue
        idx = rng.choice(len(cands), size=min(per_object, len(cands)), replace=False)
        out += [cands[i] for i in sorted(idx)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", default="lm", choices=list(DATASETS.keys()))
    ap.add_argument("--objects", default="")
    ap.add_argument("--frames-per-object", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rot-bins", default="15-30,30-60,60-90,90-180")
    ap.add_argument("--roll-bins", type=int, default=12)
    ap.add_argument("--beam-seeds", type=int, default=8)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--out", default="")
    ap.add_argument("--export-only", action="store_true", help="only write observation/T0 packages for baselines")
    ap.add_argument("--sanity-gt", action="store_true", help="diagnostic: IoU of GT-pose splat vs observed mask")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--from-export", action="store_true",
                    help="use the exported observation.npz/protocol.json packages in --out instead of the raw BOP test folders")
    ap.add_argument("--init-from", default="none", choices=["none", "bins", "vggt"],
                    help="none: full method (stage I-IV); bins: stages II-IV from the exported T0 bins; "
                         "vggt: stages II-IV from the VGGT initialisation in baseline_vggt_init.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.selftest:
        return selftest(device)

    ds = args.dataset
    out_root = Path(args.out) if args.out else ROOT / "bop6dof_results" / ds
    out_root.mkdir(parents=True, exist_ok=True)
    objects = [int(x) for x in args.objects.split(",") if x.strip()]
    if not objects:
        objects = sorted({o for (_, _, o) in list_test_targets(ds)})
    if args.from_export:
        frames = select_frames_from_export(out_root)
        if objects:
            frames = [f for f in frames if f[2] in objects]
    else:
        frames = select_frames(ds, objects, args.frames_per_object, args.seed)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    bins = [tuple(float(x) for x in b.split("-")) for b in args.rot_bins.split(",")]
    rng = np.random.default_rng(args.seed + 1)
    print(f"[{ds}] {len(frames)} frames, objects={objects}, rot bins={bins}")

    results_path = out_root / ("results.jsonl" if args.init_from == "none" else f"results_init_{args.init_from}.jsonl")
    vggt_inits: dict[str, dict] = {}
    if args.init_from == "vggt":
        vp = out_root / "baseline_vggt_init.jsonl"
        if vp.exists():
            for line in vp.read_text(encoding="utf-8").splitlines():
                try:
                    rec_v = json.loads(line)
                    vggt_inits[rec_v["tag"]] = rec_v
                except Exception:
                    pass
        print(f"[init] {len(vggt_inits)} VGGT initialisations available")
    done = set()
    done_count: dict[tuple, int] = {}
    need_per_frame = len(args.rot_bins.split(",")) if args.init_from == "bins" else 1
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                k_ = (rec["scene_id"], rec["im_id"], rec["obj_id"])
                done_count[k_] = done_count.get(k_, 0) + 1
                if done_count[k_] >= need_per_frame:
                    done.add(k_)
            except Exception:
                pass

    model_cache: dict[int, tuple] = {}
    for (scene_id, im_id, obj_id) in frames:
        key = (scene_id, im_id, obj_id)
        tag = f"s{scene_id:02d}_im{im_id:04d}_o{obj_id:02d}"
        fdir = out_root / tag
        fdir.mkdir(parents=True, exist_ok=True)
        try:
            fr = load_frame_from_export(fdir) if args.from_export else load_frame(ds, scene_id, im_id, obj_id)
            if obj_id not in model_cache:
                model_cache[obj_id] = load_model(ds, obj_id, device)
            mesh, info, pts = model_cache[obj_id]
            verts = mesh.verts.cpu().numpy().astype(np.float64)
            centre = verts.mean(axis=0)
            # work with the mesh centred at its centroid; GT shifted accordingly
            mesh_c = exp.MeshData(verts=mesh.verts - torch.from_numpy(centre).float().to(device), faces=mesh.faces,
                                  verts_normals=mesh.verts_normals, verts_colors=mesh.verts_colors)
            mesh_r = float(np.max(np.linalg.norm(verts - centre, axis=1)))
            R_gt, t_gt_c = fr.R_gt, fr.R_gt @ centre + fr.t_gt
            obs = make_virtual_obs(fr, size=args.size)
            syms = symmetry_transforms(info)

            # protocol: initial poses per rotation bin (exported; our method does not read them)
            inits = []
            if args.from_export:
                inits = json.loads((fdir / "protocol.json").read_text(encoding="utf-8")).get("inits", [])
            if not inits:
                for b in bins:
                    R0, t0, rot_err = perturb_pose(R_gt, t_gt_c, b, rng)
                    inits.append({"bin": f"{int(b[0])}-{int(b[1])}", "R0": R0.tolist(), "t0_centred": t0.tolist(),
                                  "t0": (t0 - R0 @ centre).tolist(), "rot_err_deg": rot_err})
            if not args.from_export:
                np.savez_compressed(fdir / "observation.npz", K=fr.K, mask=fr.mask, depth=fr.depth, rgb=(fr.rgb * 255).astype(np.uint8),
                                    K_v=obs.K_v, R_v=obs.R_v, mask_v=obs.mask, depth_v=obs.depth, rgb_v=(obs.rgb * 255).astype(np.uint8))
            proto_old = {}
            if (fdir / "protocol.json").exists():
                try:
                    proto_old = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
                except Exception:
                    proto_old = {}
            proto_old.update({
                "dataset": ds, "scene_id": scene_id, "im_id": im_id, "obj_id": obj_id, "inst_idx": fr.inst_idx,
                "mesh_centre_obj": centre.tolist(), "mesh_r": mesh_r, "diameter": info["diameter"],
                "gt_R": fr.R_gt.tolist(), "gt_t": fr.t_gt.tolist(), "inits": inits, "fovy_virtual": obs.fovy_deg,
            })
            (fdir / "protocol.json").write_text(json.dumps(proto_old, indent=2), encoding="utf-8")
            if args.export_only:
                print(f"[export] {tag}")
                continue
            if key in done:
                print(f"[skip] {tag} done")
                continue
            if args.sanity_gt:
                R_gt_v = obs.R_v @ R_gt
                t_gt_v = obs.R_v @ t_gt_c
                m_s, _ = splat_render(verts - centre, mesh.faces.cpu().numpy(), obs.K_v, R_gt_v, t_gt_v, obs.size, obs.size)
                print(f"  [sanity] IoU(GT splat, warped mask)={iou(m_s, obs.mask):.3f}")

            if args.init_from == "none":
                init_list = [None]
            elif args.init_from == "bins":
                init_list = [(i["bin"], np.array(i["R0"]), np.array(i["t0_centred"]), i["rot_err_deg"]) for i in inits]
            else:
                v = vggt_inits.get(tag)
                if v is None:
                    print(f"[skip] {tag}: no VGGT initialisation")
                    continue
                R0v, t0v = np.array(v["R_est"]), np.array(v["t_est"])
                init_list = [("vggt", R0v, R0v @ centre + t0v, float(v["e_R_deg"]))]
            for init in init_list:
                t_start = time.time()
                if init is None:
                    res = run_method(mesh_c, mesh_r, obs, device, fdir, roll_bins=args.roll_bins, beam_seeds=args.beam_seeds)
                    method_name, bin_name, init_err = "ours_v10", "none", None
                else:
                    bin_name, R0b, t0b, init_err = init
                    res = run_from_init(mesh_c, mesh_r, obs, device, fdir / f"init_{bin_name}", R0b, t0b, roll_bins=args.roll_bins)
                    method_name = "ours_refine" if args.init_from == "bins" else "vggt+ours_refine"
                R_est, t_est_c = pipeline_pose_to_cv(res["dist"], res["elev"], res["azim"], res["psi"], obs.R_v, device)
                met = compute_metrics(R_est, t_est_c, R_gt, t_gt_c, fr.K, pts - centre, info, syms)
                rec = {"dataset": ds, "scene_id": scene_id, "im_id": im_id, "obj_id": obj_id, "tag": tag, "method": method_name,
                       "bin": bin_name, "init_rot_err_deg_this": init_err,
                       "R_est": R_est.tolist(), "t_est": (t_est_c - R_est @ centre).tolist(), **met, **res,
                       "wall_s": time.time() - t_start,
                       "init_rot_err_deg": {i["bin"]: i["rot_err_deg"] for i in inits}}
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{tag}] {method_name}/{bin_name} e_R={met['e_R_deg']:.1f} deg  e_t={met['e_t_mm']:.1f} mm  ADD(-S)={met['add_s_metric']:.1f} mm "
                      f"(<0.1d: {met['add_lt_0.1d']})  AR_MSSD={met['ar_mssd']:.2f} AR_MSPD={met['ar_mspd']:.2f}  {res['t_total_s']:.0f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[error] {tag}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
