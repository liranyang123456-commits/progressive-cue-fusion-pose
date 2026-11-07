#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared, dependency-light (numpy / scipy / cv2) BOP-style pose metrics used by the
proposed-method evaluator (bop_full6dof_eval.py) and by every baseline runner, so that
all methods are scored by exactly the same code.

Metrics
-------
e_R_deg   geodesic rotation error
e_t_mm    translation error
add / adds  average distance (ADD) and symmetric ADD-S on the evaluation point set
add_lt_0.1d ADD(-S) < 0.1 diameter (ADD for asymmetric, ADD-S for symmetric objects)
mssd / mspd maximum symmetry-aware surface / projection distance (BOP definitions)
ar_mssd / ar_mspd  recall averaged over the BOP threshold ranges (5..50 % diameter; 5..50 px)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


def load_models_info(models_dir: Path) -> dict:
    with (Path(models_dir) / "models_info.json").open(encoding="utf-8") as f:
        return json.load(f)


def symmetry_transforms(info: dict, n_cont: int = 36) -> list[np.ndarray]:
    syms = [np.eye(4)]
    for s in info.get("symmetries_discrete", []):
        syms.append(np.array(s, dtype=np.float64).reshape(4, 4))
    for s in info.get("symmetries_continuous", []):
        axis = np.array(s["axis"], dtype=np.float64)
        offset = np.array(s.get("offset", [0, 0, 0]), dtype=np.float64)
        base = list(syms)
        syms = []
        for k in range(n_cont):
            ang = 2 * np.pi * k / n_cont
            Rk = cv2.Rodrigues(axis / np.linalg.norm(axis) * ang)[0]
            Tk = np.eye(4)
            Tk[:3, :3] = Rk
            Tk[:3, 3] = offset - Rk @ offset
            for b in base:
                syms.append(Tk @ b)
    return syms


def transform_pts(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ R.T + np.asarray(t).reshape(1, 3)


def project_pts(K: np.ndarray, pts_c: np.ndarray) -> np.ndarray:
    z = np.clip(pts_c[:, 2:3], 1e-6, None)
    uv = pts_c[:, :2] / z
    return uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    tr = np.clip((np.trace(R_est @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def compute_metrics(R_est, t_est, R_gt, t_gt, K, pts, info, syms=None) -> dict:
    """All arguments in the object frame used by `pts` (mm). R_*: (3,3) object->camera, t_*: (3,) mm."""
    from scipy.spatial import cKDTree

    if syms is None:
        syms = symmetry_transforms(info)
    diam = float(info["diameter"])
    R_est, R_gt = np.asarray(R_est, dtype=np.float64), np.asarray(R_gt, dtype=np.float64)
    t_est, t_gt = np.asarray(t_est, dtype=np.float64).reshape(3), np.asarray(t_gt, dtype=np.float64).reshape(3)
    e_R = rotation_error_deg(R_est, R_gt)
    e_t = float(np.linalg.norm(t_est - t_gt))
    pe = transform_pts(R_est, t_est, pts)
    pg = transform_pts(R_gt, t_gt, pts)
    add = float(np.mean(np.linalg.norm(pe - pg, axis=1)))
    adds = float(np.mean(cKDTree(pe).query(pg, k=1)[0]))
    mssd, mspd = np.inf, np.inf
    for S in syms:
        Rs = R_gt @ S[:3, :3]
        ts = R_gt @ S[:3, 3] + t_gt
        pgs = transform_pts(Rs, ts, pts)
        mssd = min(mssd, float(np.max(np.linalg.norm(pe - pgs, axis=1))))
        mspd = min(mspd, float(np.max(np.linalg.norm(project_pts(K, pe) - project_pts(K, pgs), axis=1))))
    symmetric = len(syms) > 1
    add_metric = adds if symmetric else add
    ar_mssd = float(np.mean([mssd < th * diam for th in np.arange(0.05, 0.51, 0.05)]))
    ar_mspd = float(np.mean([mspd < th for th in np.arange(5, 51, 5)]))
    return {
        "e_R_deg": e_R, "e_t_mm": e_t, "add": add, "adds": adds, "add_s_metric": add_metric,
        "add_lt_0.1d": bool(add_metric < 0.1 * diam), "mssd": mssd, "mspd": mspd,
        "ar_mssd": ar_mssd, "ar_mspd": ar_mspd, "diameter": diam, "symmetric": symmetric,
    }


def eval_points(models_eval_dir: Path, models_dir: Path, obj_id: int, max_pts: int = 3000) -> np.ndarray:
    """Evaluation point set: BOP models_eval vertices if present, else the model vertices."""
    import open3d as o3d  # optional heavy import, only used when loading meshes

    p = Path(models_eval_dir) / f"obj_{obj_id:06d}.ply"
    if not p.exists():
        p = Path(models_dir) / f"obj_{obj_id:06d}.ply"
    pts = np.asarray(o3d.io.read_triangle_mesh(str(p)).vertices, dtype=np.float64)
    if pts.shape[0] > max_pts:
        pts = pts[np.random.default_rng(0).choice(pts.shape[0], max_pts, replace=False)]
    return pts


def eval_points_trimesh(models_eval_dir: Path, models_dir: Path, obj_id: int, max_pts: int = 3000) -> np.ndarray:
    """Same as eval_points but via trimesh (for environments without open3d)."""
    import trimesh

    p = Path(models_eval_dir) / f"obj_{obj_id:06d}.ply"
    if not p.exists():
        p = Path(models_dir) / f"obj_{obj_id:06d}.ply"
    m = trimesh.load(str(p), force="mesh", process=False)
    pts = np.asarray(m.vertices, dtype=np.float64)
    if pts.shape[0] > max_pts:
        pts = pts[np.random.default_rng(0).choice(pts.shape[0], max_pts, replace=False)]
    return pts


def summarize_results(records: list[dict]) -> dict:
    """Aggregate a list of per-frame metric dicts."""
    if not records:
        return {}
    eR = np.array([r["e_R_deg"] for r in records])
    et = np.array([r["e_t_mm"] for r in records])
    return {
        "n": len(records),
        "e_R_mean": float(eR.mean()), "e_R_median": float(np.median(eR)),
        "e_t_mean": float(et.mean()), "e_t_median": float(np.median(et)),
        "sr_5deg": float(np.mean(eR <= 5)), "sr_10deg": float(np.mean(eR <= 10)), "sr_30deg": float(np.mean(eR <= 30)),
        "add_recall_0.1d": float(np.mean([bool(r["add_lt_0.1d"]) for r in records])),
        "ar_mssd": float(np.mean([r["ar_mssd"] for r in records])),
        "ar_mspd": float(np.mean([r["ar_mspd"] for r in records])),
    }
