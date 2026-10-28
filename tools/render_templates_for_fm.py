#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render canonical RGB templates of the known mesh (no observation, no ground truth) for the
foundation-model initialisation experiment, and crop the real RGB frame around the mask.

For every exported BOP frame package <root>/<tag>/ this writes
  fm_templates/tpl_XX.png     rendered views of the mesh (vertex colours, simple shading)
  fm_templates/real_crop.png  the real RGB frame cropped around the mask bounding box (square)
  fm_templates/meta.json      intrinsics of the renders/crop and object->camera poses (OpenCV, mm)
                              of every template, plus the crop offset in the original image.

Templates: 12 directions (icosahedron vertices) at 2.5 x diameter, looking at the mesh centroid,
up vector chosen per view. Rendered with the pipeline's nvdiffrast renderer at 448x448.
Run in the kaolin env:
  python tools/render_templates_for_fm.py --root bop6dof_results/lmo
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["VAL_DISABLE_TORCH_COMPILE"] = "1"
os.environ["FORCE_CODE_RUN_PARAMS"] = "0"
os.environ["VAL_DEBUG_RAST"] = "0"
with contextlib.redirect_stdout(io.StringIO()):
    import Exp_Camera_Optim_RT_Val_Beam_Search12291_Standford122 as exp  # noqa: E402

DATASETS = {
    "lm": Path(os.environ.get("BOP_ROOT", "data/bop")) / "lm" / "models",
    "lmo": Path(os.environ.get("BOP_ROOT", "data/bop")) / "lmo" / "models",
    "tless": Path(os.environ.get("BOP_ROOT", "data/bop")) / "tless" / "models_cad",
}
F_GL2CV = np.diag([1.0, -1.0, -1.0])


def icosahedron_dirs() -> np.ndarray:
    phi = (1 + 5 ** 0.5) / 2
    v = np.array([[0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi], [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                  [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]], dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def look_at_cv(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Object->camera (OpenCV: x right, y down, z forward) for a camera at cam_pos looking at target."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([1.0, 0, 0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_c2o = np.stack([x, y, z], axis=1)  # columns = camera axes in object frame
    R = R_c2o.T
    t = -R @ cam_pos
    return R, t


def render_template(mesh, R_cv, t_cv, K, S, device, rast, phong):
    """Render with the pipeline renderer: convert OpenCV pose to the pipeline's view frame."""
    R_view = F_GL2CV @ R_cv  # X_gl = F X_cv = F R X_obj + F t
    T_view = F_GL2CV @ t_cv
    Rt = torch.from_numpy(R_view).float().to(device)[None]
    Tt = torch.from_numpy(T_view).float().to(device)[None]
    with torch.no_grad():
        cams = exp.FoVPerspectiveCameras(device=device, R=Rt, T=Tt)
        rgb = phong(meshes_world=mesh, cameras=cams)[0, ..., :3].clamp(0, 1).cpu().numpy()
        fr = rast(meshes_world=mesh, R=Rt, T=Tt)
        m = (fr.pix_to_face[..., 0][0] >= 0).cpu().numpy()
    rgb = rgb[::-1]  # nvdiffrast rows are bottom-up
    m = m[::-1]
    img = (rgb * 255).astype(np.uint8)
    img[~m] = 255  # white background
    return img, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--fovy", type=float, default=35.0)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda")
    root = Path(args.root)
    frames = sorted([d for d in root.iterdir() if d.is_dir() and (d / "protocol.json").exists()])
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    S = args.size
    os.environ["VAL_FOVY_DEG"] = str(args.fovy)
    f = (S / 2) / math.tan(math.radians(args.fovy) / 2)
    K_t = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1.0]])
    cams0 = exp.FoVPerspectiveCameras(device=device)
    _, rast, phong = exp.build_renderers(S, S, cams0, faces_per_pixel_silhouette=1)
    mesh_cache: dict = {}
    dirs = icosahedron_dirs()
    for fdir in frames:
        proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
        ds, obj = proto["dataset"], int(proto["obj_id"])
        out = fdir / "fm_templates"
        if (out / "meta.json").exists():
            continue
        out.mkdir(exist_ok=True)
        if obj not in mesh_cache:
            import open3d as o3d
            m = o3d.io.read_triangle_mesh(str(DATASETS[ds] / f"obj_{obj:06d}.ply"))
            m.compute_vertex_normals()
            verts = np.asarray(m.vertices, dtype=np.float32)
            colors = np.asarray(m.vertex_colors, dtype=np.float32) if m.has_vertex_colors() else np.full_like(verts, 0.7)
            mesh_cache[obj] = (exp.MeshData(verts=torch.from_numpy(verts).to(device), faces=torch.from_numpy(np.asarray(m.triangles, dtype=np.int64)).to(device),
                                            verts_normals=torch.from_numpy(np.asarray(m.vertex_normals, dtype=np.float32)).to(device),
                                            verts_colors=torch.from_numpy(colors).to(device)), verts.mean(axis=0), float(np.max(np.linalg.norm(verts - verts.mean(axis=0), axis=1))))
        mesh, centre, r = mesh_cache[obj]
        dist = 2.5 * (2 * r)
        metas = []
        for i, d in enumerate(dirs):
            cam_pos = centre + dist * d
            up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([0, 1.0, 0])
            R, t = look_at_cv(cam_pos, centre, up)
            img, m = render_template(mesh, R, t, K_t, S, device, rast, phong)
            cv2.imwrite(str(out / f"tpl_{i:02d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            metas.append({"file": f"tpl_{i:02d}.png", "R": R.tolist(), "t": t.tolist(), "K": K_t.tolist()})
        obs = np.load(fdir / "observation.npz")
        rgb = obs["rgb"]
        ys, xs = np.where(obs["mask"] > 0)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.9) + 8
        H, W = obs["mask"].shape
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
        crop = rgb[y0:y1, x0:x1]
        crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "real_crop.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        sx, sy = S / (x1 - x0), S / (y1 - y0)
        K = obs["K"]
        K_crop = np.array([[K[0, 0] * sx, 0, (K[0, 2] - x0) * sx], [0, K[1, 1] * sy, (K[1, 2] - y0) * sy], [0, 0, 1]])
        (out / "meta.json").write_text(json.dumps({"templates": metas, "real_crop": {"file": "real_crop.png", "K": K_crop.tolist(), "box": [x0, y0, x1, y1]},
                                                   "mesh_centre": centre.tolist(), "mesh_r": r}, indent=2), encoding="utf-8")
        print(f"[templates] {fdir.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
