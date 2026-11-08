#!/usr/bin/env python3
"""
Training-free depth polish after the selection: point-to-plane ICP between the observed points (depth inside the visible
mask, back-projected with K) and points sampled on the mesh, started from a given pose (no ground truth). Correspondences
go from each observed point to its nearest model point among those facing the camera; pairs farther than a threshold
that shrinks from 0.15 to 0.02 diameters are dropped, the rest are Huber-weighted, and the linearised 6-DoF update is
solved by least squares. Applied to the poses of an existing results file and scored with the same metrics.

  python tools/icp_polish.py --src results_msf_select.jsonl --out results_msf_icp.jsonl   # both BOP samples
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_metrics import compute_metrics, eval_points_trimesh, load_models_info, symmetry_transforms  # noqa: E402

BOP_ROOT = Path(r"D:\bop")
MODELS = {"lm": ("models", "models_eval"), "lmo": ("models", "models_eval"), "tless": ("models_cad", "models_eval")}


def skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = skew(w / th)
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def fitness(R: np.ndarray, t: np.ndarray, obs: np.ndarray, mpts: np.ndarray, mnrm: np.ndarray, diam: float) -> float:
    """Share of the observed points within 0.02 diameters of the camera-facing model surface (observations only)."""
    X = mpts @ R.T + t
    front = np.einsum("ij,ij->i", mnrm @ R.T, X) < 0
    if front.sum() < 50:
        front[:] = True
    d, _ = cKDTree(X[front]).query(obs)
    return float(np.mean(d < 0.02 * diam))


def icp(R: np.ndarray, t: np.ndarray, obs: np.ndarray, mpts: np.ndarray, mnrm: np.ndarray, diam: float,
        iters: int = 30) -> tuple[np.ndarray, np.ndarray, int]:
    """Point-to-plane ICP, observed -> model; returns the refined pose and the number of inliers of the last step."""
    n_in = 0
    for it in range(iters):
        X = mpts @ R.T + t
        N = mnrm @ R.T
        front = np.einsum("ij,ij->i", N, X) < 0  # normals facing the camera
        if front.sum() < 50:
            front[:] = True
        Xf, Nf = X[front], N[front]
        d, idx = cKDTree(Xf).query(obs)
        thr = diam * (0.15 + (0.02 - 0.15) * it / max(1, iters - 1))
        keep = d < thr
        n_in = int(keep.sum())
        if n_in < 30:
            break
        p, q, n = obs[keep], Xf[idx[keep]], Nf[idx[keep]]
        r = np.einsum("ij,ij->i", q - p, n)
        delta = 0.01 * diam
        w = np.where(np.abs(r) < delta, 1.0, delta / np.maximum(np.abs(r), 1e-9))  # Huber
        A = np.hstack([np.cross(q, n), n]) * np.sqrt(w)[:, None]
        b = -r * np.sqrt(w)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        dR = so3_exp(x[:3])
        R, t = dR @ R, dR @ t + x[3:]
        if np.linalg.norm(x[:3]) < 1e-5 and np.linalg.norm(x[3:]) < 1e-3:
            break
    return R, t, n_in


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results_msf_select.jsonl")
    ap.add_argument("--out", default="results_msf_icp.jsonl")
    ap.add_argument("--roots", default="bop6dof_results,bop6dof_results_ext")
    ap.add_argument("--n-model", type=int, default=6000)
    ap.add_argument("--n-obs", type=int, default=4000)
    ap.add_argument("--no-accept", action="store_true", help="keep the ICP pose even when it fits the observed points worse")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    cache: dict = {}
    for root in [ROOT / r for r in args.roots.split(",")]:
        for ds in ("lmo", "lm", "tless"):
            src = root / ds / args.src
            if not src.exists():
                continue
            out_rows = []
            for line in src.read_text(encoding="utf-8").splitlines():
                r0 = json.loads(line)
                tag, obj = r0["tag"], int(r0["obj_id"])
                fdir = root / ds / tag
                proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
                if (ds, obj) not in cache:
                    mdir, edir = MODELS[ds]
                    mesh = trimesh.load(str(BOP_ROOT / ds / mdir / f"obj_{obj:06d}.ply"), force="mesh", process=False)
                    pts, fi = trimesh.sample.sample_surface(mesh, args.n_model, seed=0)
                    info = load_models_info(BOP_ROOT / ds / mdir)[str(obj)]
                    cache[(ds, obj)] = (np.asarray(pts, float), np.asarray(mesh.face_normals[fi], float), info,
                                        eval_points_trimesh(BOP_ROOT / ds / edir, BOP_ROOT / ds / mdir, obj))
                mpts, mnrm, info, epts = cache[(ds, obj)]
                obs = np.load(fdir / "observation.npz")
                K = np.asarray(obs["K"], float)
                z = obs["depth"].astype(float)
                vv, uu = np.nonzero((obs["mask"] > 0) & (z > 0))
                R0, t0 = np.array(r0["R_est"], float), np.array(r0["t_est"], float)
                ts = time.perf_counter()
                if len(vv) >= 50:
                    sel = rng.choice(len(vv), size=min(args.n_obs, len(vv)), replace=False)
                    zz = z[vv[sel], uu[sel]]
                    P = np.stack([(uu[sel] - K[0, 2]) * zz / K[0, 0], (vv[sel] - K[1, 2]) * zz / K[1, 1], zz], 1)
                    diam = float(info["diameter"])
                    R1, t1, n_in = icp(R0, t0, P, mpts, mnrm, diam)
                    f0, f1 = fitness(R0, t0, P, mpts, mnrm, diam), fitness(R1, t1, P, mpts, mnrm, diam)
                    if not args.no_accept and f1 <= f0:  # the polish must fit the observed points strictly better
                        R1, t1 = R0, t0
                else:  # no depth inside the mask: keep the pose
                    R1, t1, n_in, f0, f1 = R0, t0, 0, 0.0, 0.0
                dt = time.perf_counter() - ts
                met = compute_metrics(R1, t1, np.array(proto["gt_R"]), np.array(proto["gt_t"]), K, epts, info, symmetry_transforms(info))
                out_rows.append({"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj,
                                 "method": "icp_polish", "bin": "none", "source": args.src, "R_est": R1.tolist(), "t_est": t1.tolist(),
                                 "n_inliers": n_in, "fitness_before": f0, "fitness_after": f1, "accepted": bool(R1 is not R0),
                                 "t_icp_s": dt, "t_total_s": float(r0.get("t_total_s") or 0.0) + dt, **met})
            (root / ds / args.out).write_text("".join(json.dumps(r) + "\n" for r in out_rows), encoding="utf-8")
            print(f"{root.name}/{ds}: {len(out_rows)} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
