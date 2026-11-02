"""Numerical checks of the SDFR port (run in the `sdfr` env):
  1. so3_exp_map against scipy Rotation.from_rotvec; 2. so3_log_map round trip against scipy as_rotvec;
  3. farthest point sampling against a brute-force reference; 4. the libigl signed distance used instead of the
  mesh2sdf CUDA kernel against trimesh's independent implementation, on near-surface points of a BOP model.
Usage: python tools/sdfr_port/verify_port.py"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pytorch3d.ops import sample_farthest_points  # noqa: E402
from pytorch3d.transforms import so3_exp_map, so3_log_map  # noqa: E402

rng = np.random.default_rng(0)
axis = rng.normal(size=(2000, 3)); axis /= np.linalg.norm(axis, axis=1, keepdims=True)
ang = rng.uniform(0.0, np.radians(179.0), size=2000)
rv = axis * ang[:, None]
R_ref = Rotation.from_rotvec(rv).as_matrix()
R_p3d = so3_exp_map(torch.from_numpy(rv)).numpy()
print(f"1. so3_exp_map vs scipy: max |dR| = {np.abs(R_p3d - R_ref).max():.2e}")
lg = so3_log_map(torch.from_numpy(R_ref)).numpy()
print(f"2. so3_log_map vs scipy (angles <= 179 deg): max |d rotvec| = {np.abs(lg - rv).max():.2e}")

pts = torch.from_numpy(rng.normal(size=(3, 800, 3)))
sel, idx = sample_farthest_points(pts, K=150)
ok = True
for b in range(3):
    P = pts[b].numpy(); ref = [0]; d = np.full(len(P), np.inf)
    for _ in range(149):
        d = np.minimum(d, ((P - P[ref[-1]]) ** 2).sum(1)); ref.append(int(np.argmax(d)))
    ok &= np.array_equal(np.array(ref), idx[b].numpy())
print(f"3. farthest point sampling identical to brute force: {ok}")

import igl
import trimesh
mesh = trimesh.load(r"D:\bop\lm\models_eval\obj_000001.ply", force="mesh")
V, F = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
surf, _ = trimesh.sample.sample_surface(mesh, 3000, seed=0)
q = surf + rng.normal(scale=0.02 * np.linalg.norm(V.max(0) - V.min(0)), size=surf.shape)
S = igl.signed_distance(q, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]
T = -trimesh.proximity.signed_distance(mesh, q)  # trimesh: positive inside
print(f"4. SDF libigl vs trimesh on {len(q)} near-surface points: watertight={mesh.is_watertight}, "
      f"max |d| = {np.abs(np.abs(S) - np.abs(T)).max():.3e} mm, sign agreement = {np.mean(np.sign(S) == np.sign(T)):.4f}")
