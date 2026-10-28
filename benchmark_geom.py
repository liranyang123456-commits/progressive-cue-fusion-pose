"""Geometry metrics without PyTorch3D (safe after P3D subprocess)."""
from __future__ import annotations

import math

import numpy as np


def rot_err_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    tr = float(np.trace(R_est @ R_gt.T))
    tr = max(-1.0, min(3.0, tr))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def look_at_view_R_np(
    dist: float,
    elev_deg: float,
    azim_deg: float,
    at: np.ndarray | None = None,
) -> np.ndarray:
    """World-to-view R (rows = axes), matching PyTorch3D look_at_view_transform at=origin."""
    at = np.zeros(3, dtype=np.float64) if at is None else np.asarray(at, dtype=np.float64).reshape(3)
    elev = math.radians(float(elev_deg))
    azim = math.radians(float(azim_deg))
    d = float(dist)
    x = d * math.cos(elev) * math.sin(azim)
    y = d * math.sin(elev)
    z = d * math.cos(elev) * math.cos(azim)
    cam = np.array([x, y, z], dtype=np.float64) + at
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = _normalize(cam - at)
    x_axis = _normalize(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=0)


def look_at_cam_pos_R_np(cam_pos: np.ndarray, at: np.ndarray | None = None) -> np.ndarray:
    """R from camera world position (PyTorch3D look_at_rotation, target at origin)."""
    at = np.zeros(3, dtype=np.float64) if at is None else np.asarray(at, dtype=np.float64).reshape(3)
    cam = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = _normalize(cam - at)
    x_axis = _normalize(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=0)


def cam_pos_from_spherical_np(
    dist: float,
    elev_deg: float,
    azim_deg: float,
    at: np.ndarray | None = None,
) -> np.ndarray:
    at = np.zeros(3, dtype=np.float64) if at is None else np.asarray(at, dtype=np.float64).reshape(3)
    elev = math.radians(float(elev_deg))
    azim = math.radians(float(azim_deg))
    d = float(dist)
    offset = np.array(
        [
            d * math.cos(elev) * math.sin(azim),
            d * math.sin(elev),
            d * math.cos(elev) * math.cos(azim),
        ],
        dtype=np.float64,
    )
    return at + offset


def wrap_deg_180(deg: float) -> float:
    x = (float(deg) + 180.0) % 360.0 - 180.0
    return 180.0 if x == -180.0 else x


def view_dir_err_deg(cam_gt: np.ndarray, cam_est: np.ndarray, at: np.ndarray | None = None) -> float:
    at = np.zeros(3, dtype=np.float64) if at is None else np.asarray(at, dtype=np.float64).reshape(3)
    vg = _normalize(cam_gt - at)
    ve = _normalize(cam_est - at)
    c = float(np.clip(np.dot(vg, ve), -1.0, 1.0))
    return math.degrees(math.acos(c))


def full_pose_metrics_row(row: dict) -> dict[str, float]:
    """Unified metrics aligned with pose_metrics_summary.csv (nvdiffrast + P3D rows)."""
    at = np.zeros(3, dtype=np.float64)
    cam_gt = cam_pos_from_spherical_np(row["gt_dist"], row["gt_elev_deg"], row["gt_azim_deg"], at)
    if "est_cam_x" in row:
        cam_est = np.array([row["est_cam_x"], row["est_cam_y"], row["est_cam_z"]], dtype=np.float64)
        R_est = look_at_cam_pos_R_np(cam_est, at)
        off = cam_est - at
        de = float(np.linalg.norm(off))
        ee = math.degrees(math.asin(np.clip(off[1] / max(de, 1e-12), -1.0, 1.0)))
        ae = math.degrees(math.atan2(off[0], off[2]))
    else:
        cam_est = cam_pos_from_spherical_np(row["est_dist"], row["est_elev_deg"], row["est_azim_deg"], at)
        R_est = look_at_view_R_np(row["est_dist"], row["est_elev_deg"], row["est_azim_deg"], at)
        de, ee, ae = float(row["est_dist"]), float(row["est_elev_deg"]), float(row["est_azim_deg"])

    R_gt = look_at_view_R_np(row["gt_dist"], row["gt_elev_deg"], row["gt_azim_deg"], at)
    dg, eg, ag = float(row["gt_dist"]), float(row["gt_elev_deg"]), float(row["gt_azim_deg"])

    return {
        "rot_err_deg": rot_err_deg(R_est, R_gt),
        "cam_center_err": float(np.linalg.norm(cam_est - cam_gt)),
        "view_dir_err_deg": view_dir_err_deg(cam_gt, cam_est, at),
        "dist_err": abs(de - dg),
        "elev_err_deg": abs(ee - eg),
        "azim_err_deg": abs(wrap_deg_180(ae - ag)),
    }


def rot_err_from_p3d_row(row: dict) -> float:
    return full_pose_metrics_row(row)["rot_err_deg"]
