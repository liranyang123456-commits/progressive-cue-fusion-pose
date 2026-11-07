#!/usr/bin/env python3
"""
Failure modes of the multi-source selection on the BOP frames (symmetry-aware): for every frame whose selected pose
misses SR@10, the residual rotation R_gt^T R_sel in the object frame (angle, alignment of its axis with the principal
axes of the mesh) and the error left after composing the selected pose with discrete object-frame rotations
(180 deg about each principal axis; +-90 deg about each) -- i.e. whether a cheap set of re-seeded hypotheses would
place a start inside the local basin of stages II-IV.

  python tools/analyze_failure_modes.py [--roots bop6dof_results,bop6dof_results_ext]
Prints the table; writes benchmark_results/failure_modes.json.
"""
from __future__ import annotations

import os

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp_bop", Path(__file__).resolve().parent / "export_bop_results.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)
eb.BOP_ROOT = Path(os.environ.get("BOP_ROOT", "data/bop"))
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from bop_metrics import eval_points_trimesh  # noqa: E402

SRC = {"ours": "results.jsonl", "pi3": "results_init_pi3.jsonl", "vggt": "results_init_vggt.jsonl",
       "wid3r": "results_init_wid3r.jsonl", "cam3r": "results_init_cam3r.jsonl"}
MODELS = {"lm": ("models_eval", "models"), "lmo": ("models_eval", "models"), "tless": ("models_eval", "models_cad")}


def rodrigues(axis: np.ndarray, deg: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = np.deg2rad(deg)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    th = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.linalg.norm(w) < 1e-6:  # theta ~ 0 or ~ 180: axis from the symmetric part
        M = (R + np.eye(3)) / 2
        w = M[:, int(np.argmax(np.diag(M)))]
    return w / (np.linalg.norm(w) + 1e-12), th


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="bop6dof_results,bop6dof_results_ext")
    args = ap.parse_args()
    axes_cache: dict = {}
    rows = []
    for root in [ROOT / r for r in args.roots.split(",")]:
        for ds in ("lmo", "lm", "tless"):
            eb.RES = root
            src = {}
            for k, fn in SRC.items():
                recs = [r for r in eb.read_jsonl(root / ds / fn) if "e_R_deg" in r and "loss" in r and r.get("R_est")]
                eb.add_sym_error(ds, recs)
                src[k] = {r["tag"]: r for r in recs}
            for tag, r0 in src["ours"].items():
                cands = [src[k][tag] for k in SRC if tag in src[k]]
                sel = min(cands, key=lambda r: float(r["loss"]))
                oracle = min(float(r["e_R_sym_deg"]) for r in cands)
                Rg = eb.gt_rot(ds, tag)
                obj = int(sel["obj_id"])
                if (ds, obj) not in axes_cache:
                    me, mo = MODELS[ds]
                    P = eval_points_trimesh(eb.BOP_ROOT / ds / me, eb.BOP_ROOT / ds / mo, obj)
                    P = P - P.mean(0)
                    _, _, Vt = np.linalg.svd(P, full_matrices=False)
                    axes_cache[(ds, obj)] = Vt  # rows = principal axes (object frame), largest extent first
                V = axes_cache[(ds, obj)]
                Rs = np.array(sel["R_est"], dtype=np.float64).reshape(3, 3)
                syms = eb.sym_rots(ds, obj)
                errs = [eb.rotation_error_deg(Rs, Rg @ S) for S in syms]
                S = syms[int(np.argmin(errs))]
                ax, th = axis_angle((Rg @ S).T @ Rs)
                align = float(np.max(np.abs(V @ ax)))
                flips = {}
                for i in range(3):
                    for deg in (180.0, 90.0, -90.0):
                        F = rodrigues(V[i], deg)
                        flips[f"a{i}_{int(deg)}"] = float(min(eb.rotation_error_deg(Rs @ F, Rg @ S2) for S2 in syms))
                rows.append({"root": root.name, "ds": ds, "tag": tag, "obj": obj, "sel_src": next(k for k in SRC if tag in src[k] and src[k][tag] is sel),
                             "e_sel": float(sel["e_R_sym_deg"]), "oracle": oracle, "theta": th, "axis_align": align,
                             "flip180_best": min(v for k, v in flips.items() if k.endswith("_180")),
                             "flip_any_best": min(flips.values()), "flips": flips})
    n = len(rows)
    fail = [r for r in rows if r["e_sel"] > 10]
    print(f"frames {n}  selected SR@10 {100 * (1 - len(fail) / n):.1f}%  oracle SR@10 {100 * np.mean([r['oracle'] <= 10 for r in rows]):.1f}%")
    print(f"failures {len(fail)}: residual angle median {np.median([r['theta'] for r in fail]):.0f} deg; "
          f">150 deg {sum(r['theta'] > 150 for r in fail)}; 60-150 {sum(60 < r['theta'] <= 150 for r in fail)}; <=60 {sum(r['theta'] <= 60 for r in fail)}")
    print(f"  axis within 25 deg of a principal axis: {sum(r['axis_align'] > np.cos(np.deg2rad(25)) for r in fail)}")
    for thr in (15, 30, 45):
        a = sum(r["flip180_best"] <= thr for r in fail)
        b = sum(r["flip_any_best"] <= thr for r in fail)
        print(f"  after the best 180-deg principal-axis flip <= {thr} deg: {a}/{len(fail)};  any of 9 discrete rotations: {b}/{len(fail)}")
    for ds in ("lmo", "lm", "tless"):
        f = [r for r in fail if r["ds"] == ds]
        t = [r for r in rows if r["ds"] == ds]
        if t:
            print(f"  {ds}: {len(f)}/{len(t)} failures; flip180<=30: {sum(r['flip180_best'] <= 30 for r in f)}; any<=30: {sum(r['flip_any_best'] <= 30 for r in f)}")
    out = ROOT / "benchmark_results" / "failure_modes.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
