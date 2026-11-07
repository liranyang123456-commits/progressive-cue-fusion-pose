"""
NumPy replacement for FoundationPose's C++ extension `mycpp` (pybind11 + Eigen), which only provides
cluster_poses(). Same greedy rule as mycpp/src/app/pybind_api.cpp: keep a pose unless, for some kept
pose within `dist_diff` metres, some symmetry transform brings it within `angle_diff` degrees
(geodesic distance arccos((tr(R1 R2^T) - 1) / 2), cosine clamped to [-1, 1]).
"""
import numpy as np


def cluster_poses(angle_diff, dist_diff, poses_in, symmetry_tfs):
    poses_in = np.asarray(poses_in, dtype=np.float32)
    tfs = np.asarray(symmetry_tfs, dtype=np.float32)
    thr = float(angle_diff) / 180.0 * np.pi
    print(f"num original candidates = {len(poses_in)}")
    kept = [poses_in[0]]
    for cur in poses_in[1:]:
        cur_sym = cur[None] @ tfs  # (S,4,4)
        isnew = True
        for cl in kept:
            if np.linalg.norm(cl[:3, 3] - cur[:3, 3]) >= dist_diff:
                continue
            cos = (np.einsum("sij,ij->s", cur_sym[:, :3, :3], cl[:3, :3]) - 1.0) / 2.0
            if np.any(np.arccos(np.clip(cos, -1.0, 1.0)) < thr):
                isnew = False
                break
        if isnew:
            kept.append(cur)
    print(f"num of pose after clustering: {len(kept)}")
    return [p for p in kept]
