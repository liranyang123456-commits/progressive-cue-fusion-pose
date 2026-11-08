#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Candidate selection by depth-ICP fitness instead of the rendered loss.

The released pipeline selects the lowest-RECORDED-loss candidate and then polishes it with point-to-plane ICP
(tools/icp_polish.py). Re-scoring candidates offline with the rendered loss is not reproducible (the loss scale
is calibrated at run time), but the ICP fitness -- the fraction of observed masked depth points lying within
0.02 diameter of the visible model surface -- is computed in the original camera on the same point sample for
every candidate, so it is strictly comparable across candidates and needs no renderer.

Variants written per dataset directory (all ground-truth free):
  results_msf_fit_sel.jsonl    every candidate ICP-polished (accepted only if fitness improves), then the
                               candidate with the highest post-ICP fitness is kept; ties within --tie keep
                               the candidate the released selection chose.
Also reports a rescue variant (no file): keep the released select+ICP pose unless some other candidate beats
its fitness by more than --margin.

  python tools/icp_fitness_select.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_metrics import compute_metrics, load_models_info, rotation_error_deg, symmetry_transforms  # noqa: E402

spec = importlib.util.spec_from_file_location("icp_polish", Path(__file__).resolve().parent / "icp_polish.py")
ip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ip)

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


def sym_err(R_est, R_gt, syms) -> float:
    return min(rotation_error_deg(np.asarray(R_est, float), R_gt @ S[:3, :3]) for S in syms)


def _eval_pts(cache: dict, ds: str, edir: str, mdir: str, obj: int) -> np.ndarray:
    key = (ds, obj)
    if key not in cache:
        from bop_metrics import eval_points_trimesh
        cache[key] = eval_points_trimesh(BOP_ROOT / ds / edir, BOP_ROOT / ds / mdir, obj)
    return cache[key]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="bop6dof_results,bop6dof_results_ext")
    ap.add_argument("--tie", type=float, default=0.02, help="fitness tie band: prefer released choice within this")
    ap.add_argument("--tie-scan", default="0,0.01,0.02,0.05", help="additional tie values reported (no files)")
    ap.add_argument("--margin", type=float, default=0.05, help="rescue margin over the released pose's fitness")
    ap.add_argument("--extra-sources", default="", help="comma-separated jsonl files (e.g. baseline_wapr_full.jsonl) whose "
                    "per-frame pose joins the candidate pool; their standalone success is reported as the paired baseline")
    ap.add_argument("--out-suffix", default="", help="suffix for the output jsonl (protects the main result file)")
    ap.add_argument("--n-model", type=int, default=6000)
    ap.add_argument("--n-obs", type=int, default=4000)
    args = ap.parse_args()
    extra_srcs = tuple(x.strip() for x in args.extra_sources.split(",") if x.strip())
    rng = np.random.default_rng(0)
    eval_pts_cache: dict[tuple, np.ndarray] = {}
    ties = [float(x) for x in args.tie_scan.split(",") if x.strip() != ""]
    tot = {"n": 0, "icp_ok": 0, "fit_ok": 0, "rsc_ok": 0,
           "fit_gain": 0, "fit_loss": 0, "rsc_gain": 0, "rsc_loss": 0,
           "ex_ok": 0, "ex_n": 0, "mixgain": 0, "mixloss": 0}
    tie_ok = {tv: 0 for tv in ties}
    n_skipped, skipped_ok = 0, 0  # frames without enough masked depth: keep the released pose

    for root in resolve_roots(args.roots):
        for ds in ("lmo", "lm", "tless"):
            dsdir = root / ds
            if not (dsdir / "results.jsonl").exists():
                continue
            mdir, edir = MODELS[ds]
            minfo = load_models_info(BOP_ROOT / ds / mdir)
            cands: dict[str, list[dict]] = {}
            for fn in SRC:
                for r in read_jsonl(dsdir / fn):
                    if r.get("R_est") and "loss" in r:
                        r["_src"] = fn
                        cands.setdefault(r["tag"], []).append(r)
            extra_pose: dict[str, dict] = {}
            for fn in extra_srcs:
                for r in read_jsonl(dsdir / fn):
                    if r.get("R_est"):
                        r["_src"] = fn
                        r.setdefault("loss", float("inf"))  # the rendered loss never ranked these
                        extra_pose[r["tag"]] = r
                        if r["tag"] in cands:
                            cands[r["tag"]].append(r)
            base_sel = {r["tag"]: r for r in read_jsonl(dsdir / "results_msf_select.jsonl")}
            base_icp = {r["tag"]: r for r in read_jsonl(dsdir / "results_msf_icp.jsonl")}
            icp_cache: dict[int, tuple] = {}
            rows_fit = []
            n_ds = {"icp": 0, "fit": 0, "rsc": 0, "n": 0}
            for tag in sorted(cands):
                fdir = dsdir / tag
                if not (fdir / "observation.npz").exists() or tag not in base_icp:
                    continue
                proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
                obj = int(proto["obj_id"])
                info = minfo[str(obj)]
                syms = symmetry_transforms(info)
                npz = np.load(fdir / "observation.npz")
                depth, mask, K = npz["depth"].astype(float), npz["mask"] > 0, npz["K"].astype(float)
                R_gt = np.array(proto["gt_R"], float)
                t_gt = np.array(proto["gt_t"], float).reshape(3)
                if obj not in icp_cache:
                    tm = trimesh.load(str(BOP_ROOT / ds / mdir / f"obj_{obj:06d}.ply"), force="mesh", process=False)
                    mp, fidx = trimesh.sample.sample_surface(tm, args.n_model, seed=0)
                    icp_cache[obj] = (np.asarray(mp, float), np.asarray(tm.face_normals[fidx], float), float(info["diameter"]))
                mpts, mnrm, diam = icp_cache[obj]
                vv, uu = np.nonzero(mask & (depth > 0))
                if len(vv) < 50:
                    # no usable masked depth: the fitness selector is undefined, keep the released pose
                    rel = base_icp[tag]
                    ok_rel = sym_err(np.array(rel["R_est"], float), R_gt, syms) <= 10
                    n_skipped += 1
                    skipped_ok += int(ok_rel)
                    rows_fit.append({**{k: rel[k] for k in ("tag", "dataset", "scene_id", "im_id", "obj_id") if k in rel},
                                     "method": "msf_fit_sel", "bin": "none", "source": "released_no_depth",
                                     "icp_accepted": False, "fitness": float("nan"), "fitness_released": float("nan"),
                                     "oracle_ok": bool(ok_rel),
                                     "n_candidates": len(cands[tag]), "loss": rel.get("loss", float("nan")),
                                     "R_est": rel["R_est"], "t_est": rel["t_est"],
                                     **compute_metrics(np.array(rel["R_est"], float), np.array(rel["t_est"], float),
                                                       R_gt, t_gt, K,
                                                       _eval_pts(eval_pts_cache, ds, edir, mdir, obj), info, syms),
                                     "e_R_sym_deg": sym_err(np.array(rel["R_est"], float), R_gt, syms)})
                    continue
                sel = rng.choice(len(vv), size=min(args.n_obs, len(vv)), replace=False)
                zz = depth[vv[sel], uu[sel]]
                P = np.stack([(uu[sel] - K[0, 2]) * zz / K[0, 0], (vv[sel] - K[1, 2]) * zz / K[1, 1], zz], 1)

                # the pose kept by the released pipeline (select + ICP-after)
                rel = base_icp[tag]
                R_rel, t_rel = np.array(rel["R_est"], float), np.array(rel["t_est"], float)
                f_rel = ip.fitness(R_rel, t_rel, P, mpts, mnrm, diam)
                sel_rec = base_sel.get(tag, {})

                polished = []  # (fitness_after, R, t, src, accepted, is_released_choice)
                for rec in cands[tag]:
                    R0, t0 = np.array(rec["R_est"], float), np.array(rec["t_est"], float)
                    R_i, t_i, _ = ip.icp(R0, t0, P, mpts, mnrm, diam)
                    f0 = ip.fitness(R0, t0, P, mpts, mnrm, diam)
                    f1 = ip.fitness(R_i, t_i, P, mpts, mnrm, diam)
                    if f1 > f0:
                        R1, t1, acc, fa = R_i, t_i, True, f1
                    else:
                        R1, t1, acc, fa = R0, t0, False, f0
                    is_rel = bool(sel_rec) and rec["_src"] == sel_rec.get("source") and abs(rec["loss"] - sel_rec.get("loss", -1)) < 1e-9
                    polished.append((fa, R1, t1, rec["_src"], acc, is_rel))

                # fitness selection with a tie band that prefers the released choice
                fmax = max(p[0] for p in polished)
                in_tie = [p for p in polished if p[0] >= fmax - args.tie]
                pick = next((p for p in in_tie if p[5]), None) or max(in_tie, key=lambda p: p[0])
                for tv in ties:
                    cand_t = [p for p in polished if p[0] >= fmax - tv]
                    pk = next((p for p in cand_t if p[5]), None) or max(cand_t, key=lambda p: p[0])
                    tie_ok[tv] += int(sym_err(pk[1], R_gt, syms) <= 10)

                # rescue: keep the released pose unless someone beats its fitness clearly
                best = max(polished, key=lambda p: p[0])
                rsc = (f_rel, R_rel, t_rel, "released", True, True) if best[0] <= f_rel + args.margin else best

                ok_icp = sym_err(R_rel, R_gt, syms) <= 10
                ok_fit = sym_err(pick[1], R_gt, syms) <= 10
                ok_rsc = sym_err(rsc[1], R_gt, syms) <= 10
                if extra_srcs and tag in extra_pose:
                    ok_ex = sym_err(np.array(extra_pose[tag]["R_est"], float), R_gt, syms) <= 10
                    tot["ex_n"] += 1
                    tot["ex_ok"] += int(ok_ex)
                    tot["mixgain"] += int(ok_fit and not ok_ex)
                    tot["mixloss"] += int(ok_ex and not ok_fit)
                for k, v in (("icp", ok_icp), ("fit", ok_fit), ("rsc", ok_rsc)):
                    n_ds[k] += int(v)
                n_ds["n"] += 1
                tot["n"] += 1
                tot["icp_ok"] += ok_icp
                tot["fit_ok"] += ok_fit
                tot["rsc_ok"] += ok_rsc
                tot["fit_gain"] += int(ok_fit and not ok_icp)
                tot["fit_loss"] += int(ok_icp and not ok_fit)
                tot["rsc_gain"] += int(ok_rsc and not ok_icp)
                tot["rsc_loss"] += int(ok_icp and not ok_rsc)

                fa, R1, t1, src, acc, _ = pick
                oracle_ok = any(sym_err(p[1], R_gt, syms) <= 10 for p in polished)  # diagnostic only, not used by the selection
                met = compute_metrics(R1, t1, R_gt, t_gt, K, _eval_pts(eval_pts_cache, ds, edir, mdir, obj), info, syms)
                met["e_R_sym_deg"] = sym_err(R1, R_gt, syms)
                rows_fit.append({"tag": tag, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"],
                                 "obj_id": obj, "method": "msf_fit_sel", "bin": "none", "source": src,
                                 "icp_accepted": bool(acc), "fitness": fa, "fitness_released": f_rel,
                                 "oracle_ok": bool(oracle_ok), "n_candidates": len(cands[tag]), "loss": float("nan"),
                                 "R_est": np.asarray(R1).tolist(), "t_est": np.asarray(t1).tolist(), **met})

            (dsdir / f"results_msf_fit_sel{args.out_suffix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_fit), encoding="utf-8")
            if n_ds["n"]:
                print(f"== {root.name}/{ds}: N={n_ds['n']}  released(select+ICP) {100*n_ds['icp']/n_ds['n']:.1f}%  "
                      f"fitness-select {100*n_ds['fit']/n_ds['n']:.1f}%  rescue {100*n_ds['rsc']/n_ds['n']:.1f}%", flush=True)

    from math import comb
    n = max(1, tot["n"])

    def mcn(b: int, c: int) -> float:
        return 1.0 if b + c == 0 else min(1.0, 2 * sum(comb(b + c, i) for i in range(0, min(b, c) + 1)) / 2 ** (b + c))

    print(f"\n=== pooled N={tot['n']} (fitness defined): released {100*tot['icp_ok']/n:.1f}%  "
          f"fitness-select {100*tot['fit_ok']/n:.1f}% (+{tot['fit_gain']}/-{tot['fit_loss']}, p={mcn(tot['fit_gain'], tot['fit_loss']):.4f})  "
          f"rescue {100*tot['rsc_ok']/n:.1f}% (+{tot['rsc_gain']}/-{tot['rsc_loss']}, p={mcn(tot['rsc_gain'], tot['rsc_loss']):.4f})")
    print(f"=== tie-band scan (same frames): " + "  ".join(f"tie={tv:g}: {100*tie_ok[tv]/n:.1f}%" for tv in ties))
    N = tot["n"] + n_skipped
    print(f"=== full protocol N={N} (skipped {n_skipped} frames without masked depth keep the released pose): "
          f"released {100*(tot['icp_ok']+skipped_ok)/max(1,N):.1f}%  fitness-select {100*(tot['fit_ok']+skipped_ok)/max(1,N):.1f}%")
    if extra_srcs and tot["ex_n"]:
        pm = mcn(tot["mixgain"], tot["mixloss"])
        print(f"=== extra source(s) {','.join(extra_srcs)} on the paired frames (N={tot['ex_n']}): alone "
              f"{100*tot['ex_ok']/tot['ex_n']:.1f}%  mixed pool (ours + extra, per-candidate ICP + fitness) "
              f"{100*tot['fit_ok']/max(1,tot['n']):.1f}%  (+{tot['mixgain']}/-{tot['mixloss']} vs extra alone, p={pm:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
