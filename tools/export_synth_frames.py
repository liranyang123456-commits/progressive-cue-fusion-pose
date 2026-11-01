#!/usr/bin/env python3
"""
Export the synthetic Stanford benchmark views as frame packages in the format of the BOP protocol
(observation.npz + protocol.json), so that the BOP wrappers of WAPR and Diff-DOPE can be run on exactly
the observations the proposed method sees (640x480 RGB, metric depth, silhouette, fovy 60 deg).

Conventions
  * the cached poses are world-to-view in the OpenGL style of the renderer (x right, y up, camera looks
    along -z); OpenCV object-to-camera poses are obtained with F = diag(1, -1, -1);
  * every mesh is re-centred at its look-at centre and scaled to a bounding radius of 100 mm, because both
    methods expect metric object sizes; a uniform similarity changes neither rotation errors nor the
    radius-normalised distance errors reported in the tables;
  * the initial pose of the refiners is the benchmark's fixed initial camera (30 deg elevation, 0 deg
    azimuth, twice the bounding-box diagonal), i.e. the start of every DR baseline;
  * the ground truth is written to protocol.json for evaluation only.

  python tools/export_synth_frames.py            # all five meshes -> synth_frames/<mesh>/gt_XX/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "benchmark_runs"
OUT = ROOT / "synth_frames"
MESHES = [("bunny", 100), ("dragon", 100), ("armadillo", 50), ("happy", 50), ("lucy", 50)]
F = np.diag([1.0, -1.0, -1.0])
R_TARGET_MM = 100.0
FOVY_DEG = 60.0


def mesh_radius(ds: str, n: int) -> float:
    txt = (RUNS / f"{ds}_{n}gt" / "ours_v10" / "run.log").read_text(encoding="utf-8", errors="ignore")
    return float(re.search(r"\br=([0-9.]+)\s+radii_mults", txt).group(1))


def look_at_gl(at: np.ndarray, dist: float, elev: float, azim: float) -> np.ndarray:
    """Same construction as look_at_view_transform of the pipeline (world-to-view, OpenGL style)."""
    e, a = math.radians(elev), math.radians(azim)
    cam = at + dist * np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
    z = cam - at
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ cam
    return T


def look_at_centre(run: Path, tag: str) -> np.ndarray:
    """Look-at centre of the benchmark cameras: stored in gt_pose_info.json, or recovered from the first ground-truth
    camera of gt_poses.csv (camera centre minus dist times the spherical direction)."""
    info = run / tag / "gt_pose_info.json"
    if info.exists():
        return np.array(json.loads(info.read_text(encoding="utf-8"))["pose_metrics"]["camera_center_world"]["at"], dtype=np.float64)
    row = next(csv.DictReader((run / "gt_poses.csv").open(encoding="utf-8")))
    T = np.array([float(row[f"m{i}{j}"]) for i in range(4) for j in range(4)]).reshape(4, 4)
    cam = -T[:3, :3].T @ T[:3, 3]
    e, a = math.radians(float(row["elev_deg"])), math.radians(float(row["azim_deg"]))
    return cam - float(row["dist"]) * np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


def rot_err(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def to_cv(T_gl: np.ndarray, at: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """OpenGL world-to-view -> OpenCV object-to-camera for the re-centred, scaled mesh (mm)."""
    R = F @ T_gl[:3, :3]
    t = F @ T_gl[:3, 3]
    return R, s * (R @ at + t)


def inside_fraction(V: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray, mask: np.ndarray, flip: bool) -> float:
    P = (R @ V.T).T + t
    P = P[P[:, 2] > 1e-6]
    uv = (K @ (P / P[:, 2:3]).T).T[:, :2]
    H, W = mask.shape
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if ok.sum() < 50:
        return 0.0
    m = mask[::-1] if flip else mask
    return float(m[v[ok], u[ok]].mean())


def main() -> int:
    import trimesh

    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", default=",".join(d for d, _ in MESHES))
    args = ap.parse_args()
    want = set(args.meshes.split(","))
    for ds, n in MESHES:
        if ds not in want:
            continue
        run = RUNS / f"{ds}_{n}gt" / "ours_v10"
        r = mesh_radius(ds, n)
        s = R_TARGET_MM / r
        mesh = trimesh.load(str(run / "mesh_exports" / "mesh_sanitized_colored.ply"), force="mesh", process=False)
        tags = sorted(d.name for d in run.iterdir() if d.is_dir() and d.name.startswith("gt_"))
        at = look_at_centre(run, tags[0])
        diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        T_init = look_at_gl(at, 2.0 * diag, 30.0, 0.0)
        saved = [run / t / "init_pose_4x4.npy" for t in tags if (run / t / "init_pose_4x4.npy").exists()]
        if saved:
            dev = float(np.abs(np.load(saved[0]) - T_init).max())
            print(f"[{ds}] fixed initial pose reconstructed, max |dT| vs saved = {dev:.2e}")
            assert dev < 1e-3 * max(1.0, diag), "initial-pose reconstruction does not match the saved one"
        out_mesh = mesh.copy()
        out_mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - at) * s
        (OUT / ds).mkdir(parents=True, exist_ok=True)
        mesh_path = OUT / ds / "model_r100mm.ply"
        out_mesh.export(str(mesh_path))
        hull = out_mesh.convex_hull.vertices
        diameter = float(max(np.linalg.norm(hull - p, axis=1).max() for p in hull))
        Vs = np.asarray(out_mesh.vertices)[:: max(1, len(out_mesh.vertices) // 4000)]
        gtcsv = {row["tag"]: row for row in csv.DictReader((run / "pose_metrics_summary.csv").open(encoding="utf-8"))}
        gt_mats = {f"gt_{int(row['idx']):02d}": np.array([float(row[f"m{i}{j}"]) for i in range(4) for j in range(4)]).reshape(4, 4)
                   for row in csv.DictReader((run / "gt_poses.csv").open(encoding="utf-8"))}
        npy0 = run / tags[0] / "gt_pose_4x4.npy"
        if npy0.exists():
            print(f"[{ds}] gt_poses.csv vs gt_pose_4x4.npy: max |dT| = {float(np.abs(np.load(npy0) - gt_mats[tags[0]]).max()):.2e}")
        flip = None
        fr = []
        for tag in tags:
            refs = np.load(run / tag / "gt_refs_full.npz")
            alpha, depth = refs["alpha"], refs["depth"]
            H, W = alpha.shape
            fy = (H / 2.0) / math.tan(math.radians(FOVY_DEG) / 2.0)
            K = np.array([[fy, 0.0, W / 2.0 - 0.5], [0.0, fy, H / 2.0 - 0.5], [0.0, 0.0, 1.0]])
            R_gt, t_gt = to_cv(gt_mats[tag], at, s)
            mask = ((alpha > 0.5) & (depth > 0)).astype(np.uint8)  # silhouette and depth must agree (AA edge pixels have alpha > 0.5 but no depth)
            if flip is None:
                f0, f1 = inside_fraction(Vs, R_gt, t_gt, K, mask, False), inside_fraction(Vs, R_gt, t_gt, K, mask, True)
                flip = f1 > f0
                print(f"[{ds}] image orientation check: inside-mask fraction {f0:.3f} (as stored) / {f1:.3f} (flipped) -> flip={flip}")
            if flip:
                alpha, depth, mask = alpha[::-1], depth[::-1], mask[::-1]
            rgb = np.load(run / tag / "gt_rgb_ref.npy")
            if flip:
                rgb = rgb[::-1]
            fr.append(inside_fraction(Vs, R_gt, t_gt, K, mask, False))
            R0, t0 = to_cv(T_init, at, s)
            fdir = OUT / ds / tag
            fdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fdir / "observation.npz", K=K, rgb=np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8),
                                depth=(depth * s * mask).astype(np.float32), mask=mask * 255)
            g = gtcsv.get(tag, {})
            proto = {"dataset": ds, "tag": tag, "scene_id": 0, "im_id": int(tag.split("_")[1]), "obj_id": 1,
                     "mesh_path": str(mesh_path), "diameter_mm": diameter, "r_native": r, "scale_mm_per_unit": s,
                     "at_native": at.tolist(), "gt_R": R_gt.tolist(), "gt_t": t_gt.tolist(),
                     "gt_dist": float(g.get("gt_dist", "nan")), "gt_elev_deg": float(g.get("gt_elev_deg", "nan")),
                     "gt_azim_deg": float(g.get("gt_azim_deg", "nan")),
                     "inits": [{"bin": "fixed", "R0": R0.tolist(), "t0": t0.tolist(), "rot_err_deg": rot_err(R0, R_gt)}]}
            (fdir / "protocol.json").write_text(json.dumps(proto, indent=1), encoding="utf-8")
        print(f"[{ds}] {len(tags)} views -> {OUT / ds}; r={r:.6g} -> scale {s:.6g} mm/unit; diameter {diameter:.1f} mm; "
              f"inside-mask fraction min/median {min(fr):.3f}/{float(np.median(fr)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())