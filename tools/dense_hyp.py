#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense rotation hypotheses with an analytic translation, scored by depth-ICP fitness (no rendering, no training,
no ground truth) -- a prototype of replacing the shell search's proposal stage on sensor data.

Per frame: 42 Fibonacci-sphere viewing directions x 6 in-plane rolls = 252 rotations; the translation is the
mask-centroid ray times (median masked depth + kappa * mesh radius), the same metric depth hint as stage II
(kappa fixed to 0.57, the ICP absorbs the residual). Every hypothesis is polished by the same point-to-plane
ICP as tools/icp_polish.py (accepted only if it fits the observed points better) and the candidate with the
highest fitness is kept.

Reports per dataset and pooled: oracle coverage of the polished dense set (any hypothesis within 10 deg sym),
fitness-top1 success, and the same after merging the dense set with the existing 14-candidate pool.

  python tools/dense_hyp.py --workers 8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BOP_ROOT = Path(os.environ.get("BOP_ROOT", r"D:\bop" if Path(r"D:\bop").exists() else "data/bop"))
_ROOT_ALIAS = {"bop6dof_results_ext": "bop6dof_results_second"}


def resolve_roots(spec: str) -> list[Path]:
    """Working tree uses bop6dof_results_ext; the release tree stores the second sample as
    results/bop6dof_results_second."""
    found = []
    for raw in [r.strip() for r in spec.split(",") if r.strip()]:
        names = [raw] + ([_ROOT_ALIAS[raw]] if raw in _ROOT_ALIAS else [])
        hit = None
        for name in names:
            for cand in (ROOT / name, ROOT / "results" / name):
                if cand.exists():
                    hit = cand
                    break
            if hit:
                break
        if hit:
            found.append(hit)
    return found
SRC = ("results.jsonl", "results_init_pi3.jsonl", "results_init_vggt.jsonl",
       "results_init_wid3r.jsonl", "results_init_cam3r.jsonl", "results_init_flip180.jsonl", "results_init_flip90.jsonl")
MODELS = {"lm": ("models", "models_eval"), "lmo": ("models", "models_eval"), "tless": ("models_cad", "models_eval")}
KAPPA = 0.57
_CACHE: dict = {}


def _ip():
    if "ip" not in _CACHE:
        spec = importlib.util.spec_from_file_location("icp_polish", Path(__file__).resolve().parent / "icp_polish.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _CACHE["ip"] = m
    return _CACHE["ip"]


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def model(ds: str, obj: int):
    key = (ds, obj)
    if key not in _CACHE:
        import trimesh
        from bop_metrics import load_models_info
        mdir, _ = MODELS[ds]
        tm = trimesh.load(str(BOP_ROOT / ds / mdir / f"obj_{obj:06d}.ply"), force="mesh", process=False)
        mp, fidx = trimesh.sample.sample_surface(tm, 6000, seed=0)
        verts = np.asarray(tm.vertices, float)
        centre = verts.mean(axis=0)
        info = load_models_info(BOP_ROOT / ds / mdir)[str(obj)]
        from bop_metrics import symmetry_transforms
        _CACHE[key] = (np.asarray(mp, float), np.asarray(tm.face_normals[fidx], float), float(info["diameter"]),
                       centre, float(np.max(np.linalg.norm(verts - centre, axis=1))), symmetry_transforms(info))
    return _CACHE[key]


def dense_rotations(n_dir: int = 42, n_roll: int = 6) -> list[np.ndarray]:
    """Object-to-camera rotations: the camera looks at the object centre from n_dir Fibonacci directions, n_roll rolls."""
    out = []
    ga = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_dir):
        y = 1.0 - 2.0 * (i + 0.5) / n_dir
        r = np.sqrt(max(0.0, 1.0 - y * y))
        v = np.array([np.cos(ga * i) * r, y, np.sin(ga * i) * r])  # object -> camera direction
        z = -v  # camera z axis points at the object
        up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        yax = np.cross(z, x)
        R0 = np.stack([x, yax, z])  # rows: world(object) axes of the camera frame -> object-to-camera rotation
        for k in range(n_roll):
            ps = 2.0 * np.pi * k / n_roll
            c, s = np.cos(ps), np.sin(ps)
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            out.append(Rz @ R0)
    return out


ROTS = dense_rotations()


def sym_err(R_est, R_gt, syms) -> float:
    from bop_metrics import rotation_error_deg
    return min(rotation_error_deg(np.asarray(R_est, float), R_gt @ S[:3, :3]) for S in syms)


def work(task: tuple) -> dict | None:
    root_s, ds, tag = task
    ip = _ip()
    dsdir = ROOT / root_s / ds
    fdir = dsdir / tag
    if not (fdir / "observation.npz").exists():
        return None
    proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
    obj = int(proto["obj_id"])
    mpts, mnrm, diam, centre, r_mesh, syms = model(ds, obj)
    npz = np.load(fdir / "observation.npz")
    depth, mask, K = npz["depth"].astype(float), npz["mask"] > 0, npz["K"].astype(float)
    R_gt = np.array(proto["gt_R"], float)
    vv, uu = np.nonzero(mask & (depth > 0))
    if len(vv) < 50:
        return None
    rng = np.random.default_rng(0)
    sel = rng.choice(len(vv), size=min(4000, len(vv)), replace=False)
    zz = depth[vv[sel], uu[sel]]
    P = np.stack([(uu[sel] - K[0, 2]) * zz / K[0, 0], (vv[sel] - K[1, 2]) * zz / K[1, 1], zz], 1)

    uc, vc = float(uu.mean()), float(vv.mean())
    ray = np.array([(uc - K[0, 2]) / K[0, 0], (vc - K[1, 2]) / K[1, 1], 1.0])
    ray /= np.linalg.norm(ray)
    d_c = float(np.median(zz)) + KAPPA * r_mesh
    t_c = ray * d_c

    best_f, best_e, ora = -1.0, None, False
    for R in ROTS:
        t0 = t_c - R @ centre
        R_i, t_i, _ = ip.icp(R, t0, P, mpts, mnrm, diam)
        f0 = ip.fitness(R, t0, P, mpts, mnrm, diam)
        f1 = ip.fitness(R_i, t_i, P, mpts, mnrm, diam)
        R1, t1, f = (R_i, t_i, f1) if f1 > f0 else (R, t0, f0)
        e = sym_err(R1, R_gt, syms)
        ora = ora or e <= 10
        if f > best_f:
            best_f, best_e = f, e

    # existing pool, same ICP + fitness (for the merged number)
    pool_f, pool_e = -1.0, None
    for fn in SRC:
        for r in read_jsonl(dsdir / fn):
            if r.get("tag") != tag or not r.get("R_est"):
                continue
            R0, t0 = np.array(r["R_est"], float), np.array(r["t_est"], float)
            R_i, t_i, _ = ip.icp(R0, t0, P, mpts, mnrm, diam)
            f0 = ip.fitness(R0, t0, P, mpts, mnrm, diam)
            f1 = ip.fitness(R_i, t_i, P, mpts, mnrm, diam)
            R1, t1, f = (R_i, t_i, f1) if f1 > f0 else (R0, t0, f0)
            if f > pool_f:
                pool_f, pool_e = f, sym_err(R1, R_gt, syms)
    mrg_e = best_e if best_f >= pool_f else pool_e
    return {"root": root_s, "ds": ds, "tag": tag, "oracle": bool(ora),
            "dense_ok": bool(best_e is not None and best_e <= 10), "dense_f": best_f, "dense_e": best_e,
            "pool_ok": bool(pool_e is not None and pool_e <= 10),
            "merged_ok": bool(mrg_e is not None and mrg_e <= 10)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="bop6dof_results,bop6dof_results_ext")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    tasks = []
    for root in resolve_roots(args.roots):
        root_s = root.relative_to(ROOT).as_posix()
        for ds in ("lmo", "lm", "tless"):
            dsdir = root / ds
            if not (dsdir / "results.jsonl").exists():
                continue
            tags = sorted({r["tag"] for r in read_jsonl(dsdir / "results.jsonl") if "tag" in r})
            if args.limit:
                tags = tags[: args.limit]
            tasks += [(root_s, ds, t) for t in tags]
    print(f"{len(tasks)} frames x {len(ROTS)} rotations, {args.workers} workers", flush=True)
    rows = []
    if args.workers <= 1:
        for i, t in enumerate(tasks):
            r = work(t)
            if r:
                rows.append(r)
            print(f"[{i+1}/{len(tasks)}] {t[2]}: " + (f"dense={int(r['dense_ok'])} oracle={int(r['oracle'])} "
                  f"pool={int(r['pool_ok'])} merged={int(r['merged_ok'])}" if r else "skip"), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(work, tasks)):
                if r:
                    rows.append(r)
                if (i + 1) % 10 == 0:
                    print(f"[{i+1}/{len(tasks)}] done", flush=True)

    out_dir = ROOT / "benchmark_results"
    if not out_dir.exists() and (ROOT / "results" / "benchmark_results").exists():
        out_dir = ROOT / "results" / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dense_hyp_results.json").write_text(json.dumps(rows), encoding="utf-8")
    keys = ("dense_ok", "oracle", "pool_ok", "merged_ok")
    for root_s in {r["root"] for r in rows}:
        for ds in ("lmo", "lm", "tless"):
            sub = [r for r in rows if r["ds"] == ds and r["root"] == root_s]
            if sub:
                print(f"== {root_s}/{ds}: N={len(sub)}  " + "  ".join(f"{k} {100*np.mean([r[k] for r in sub]):.1f}%" for k in keys))
    n = len(rows)
    print(f"\n=== pooled N={n}: " + "  ".join(f"{k} {100*np.mean([r[k] for r in rows]):.1f}%" for k in keys))
    b = sum(1 for r in rows if r["merged_ok"] and not r["pool_ok"])
    c = sum(1 for r in rows if r["pool_ok"] and not r["merged_ok"])
    p = 1.0 if b + c == 0 else min(1.0, 2 * sum(comb(b + c, i) for i in range(0, min(b, c) + 1)) / 2 ** (b + c))
    print(f"=== merged vs existing pool: +{b}/-{c}, exact McNemar p={p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
