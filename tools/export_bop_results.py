#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate the BOP full-6-DoF evaluation (ours + baselines on identical frames / initial poses) into
LaTeX tables and macros for the manuscript.

  python tools/export_bop_results.py

Reads bop6dof_results/<ds>/{results.jsonl, results_init_bins.jsonl, results_init_vggt.jsonl,
baseline_wapr_refine.jsonl, baseline_wapr_full.jsonl, baseline_diffdope.jsonl, baseline_vggt_init.jsonl}
Writes manuscript/.../tables/tab_bop.tex, tab_bop_fminit.tex and bop_macros.tex (+ benchmark_results/bop_summary.json).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_metrics import load_models_info, rotation_error_deg, symmetry_transforms  # noqa: E402
RES = ROOT / "bop6dof_results"
MAN = ROOT / "manuscript" / "computational_visual_media"
TABLES = MAN / "tables"
DATASETS = [("lmo", "LM-O"), ("lm", "LM"), ("tless", "T-LESS")]
BINS = ["15-30", "30-60", "60-90", "90-180"]
BIN_KEY = {"15-30": "BinA", "30-60": "BinB", "60-90": "BinC", "90-180": "BinD"}  # LaTeX macro names cannot contain digits
RNG = np.random.default_rng(12291)
BOP_ROOT = Path(os.environ.get("BOP_ROOT", os.environ.get("BOP_ROOT", "data/bop")))
MODELS_DIR = {"lm": "models", "lmo": "models", "tless": "models_cad"}
_INFO: dict[str, dict] = {}
_SYMS: dict[tuple, list] = {}
_GT: dict[tuple, np.ndarray | None] = {}
SYM_CHECK = {"n": 0, "max_dev": 0.0}


def sym_rots(ds: str, obj_id) -> list[np.ndarray]:
    """Rotation parts of the object's symmetry transforms (models_info.json; continuous symmetries sampled at 5 deg)."""
    key = (ds, int(obj_id))
    if key not in _SYMS:
        if ds not in _INFO:
            mdir = BOP_ROOT / ds / MODELS_DIR[ds]
            _INFO[ds] = load_models_info(mdir) if (mdir / "models_info.json").exists() else {}
        info = _INFO[ds].get(str(int(obj_id)))
        _SYMS[key] = [S[:3, :3] for S in symmetry_transforms(info, n_cont=72)] if info else [np.eye(3)]
    return _SYMS[key]


def gt_rot(ds: str, tag: str):
    key = (ds, tag)
    if key not in _GT:
        pj = RES / ds / str(tag) / "protocol.json"
        _GT[key] = np.array(json.loads(pj.read_text(encoding="utf-8"))["gt_R"], dtype=np.float64).reshape(3, 3) if pj.exists() else None
    return _GT[key]


def add_sym_error(ds: str, recs: list[dict]) -> None:
    """e_R_sym_deg = min over the object's symmetry transforms of the rotation error (identity for asymmetric objects).
    The raw e_R_deg stored by the evaluators is re-derived from R_est and the protocol's gt_R as a frame-convention check."""
    for r in recs:
        r["e_R_sym_deg"] = float(r["e_R_deg"])
        if "R_est" not in r or "tag" not in r or "obj_id" not in r:
            continue
        Rg = gt_rot(ds, r["tag"])
        if Rg is None:
            continue
        try:
            Re = np.array(r["R_est"], dtype=np.float64).reshape(3, 3)
        except Exception:  # failure records of a baseline may carry no pose
            continue
        raw = rotation_error_deg(Re, Rg)
        SYM_CHECK["n"] += 1
        SYM_CHECK["max_dev"] = max(SYM_CHECK["max_dev"], abs(raw - float(r["e_R_deg"])))
        r["e_R_sym_deg"] = float(min(rotation_error_deg(Re, Rg @ S) for S in sym_rots(ds, r["obj_id"])))


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def load_all(ds: str) -> dict[str, list[dict]]:
    d = RES / ds
    groups: dict[str, list[dict]] = {}
    for name, fn in (("ours_full", "results.jsonl"), ("ours_refine", "results_init_bins.jsonl"), ("vggt_ours", "results_init_vggt.jsonl"),
                     ("ours_full_unbal", "results_unbalanced.jsonl"), ("ours_refine_unbal", "results_init_bins_unbalanced.jsonl"),
                     ("ours_wapr", "baseline_wapr_from_ours.jsonl"),
                     ("wapr_refine", "baseline_wapr_refine.jsonl"), ("wapr_full", "baseline_wapr_full.jsonl"), ("diffdope", "baseline_diffdope.jsonl"),
                     ("vggt_init", "baseline_vggt_init.jsonl"),
                     ("pi3_init", "baseline_pi3_init.jsonl"), ("pi3_ours", "results_init_pi3.jsonl"),
                     ("wid3r_init", "baseline_wid3r_init.jsonl"), ("wid3r_ours", "results_init_wid3r.jsonl"),
                     ("cam3r_init", "baseline_cam3r_init.jsonl"), ("cam3r_ours", "results_init_cam3r.jsonl"),
                     ("ours_full_occ", "results_occ.jsonl"), ("scflow2", "baseline_scflow2.jsonl"), ("sdfr", "baseline_sdfr.jsonl")):
        recs = read_jsonl(d / fn)
        # keep the last record per (tag, bin)
        uniq: dict[tuple, dict] = {}
        for r in recs:
            if "e_R_deg" not in r:
                continue
            uniq[(r.get("tag"), str(r.get("bin", "none")))] = r
        groups[name] = list(uniq.values())
        add_sym_error(ds, groups[name])
    return groups


def summarize(recs: list[dict]) -> dict:
    if not recs:
        return {}
    eR = np.array([float(r["e_R_deg"]) for r in recs])
    eRs = np.array([float(r.get("e_R_sym_deg", r["e_R_deg"])) for r in recs])
    et = np.array([float(r["e_t_mm"]) for r in recs])
    add = np.array([bool(r["add_lt_0.1d"]) for r in recs])
    ar_s = np.array([float(r["ar_mssd"]) for r in recs])
    ar_p = np.array([float(r["ar_mspd"]) for r in recs])
    tt = np.array([float(r.get("t_total_s", r.get("wall_s", np.nan))) for r in recs])
    # bootstrap CI for SR@10
    boots = []
    for _ in range(2000):
        idx = RNG.integers(0, len(eR), len(eR))
        boots.append(np.mean(eR[idx] <= 10))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    boots_s = []
    for _ in range(2000):
        idx = RNG.integers(0, len(eRs), len(eRs))
        boots_s.append(np.mean(eRs[idx] <= 10))
    lo_s, hi_s = np.percentile(boots_s, [2.5, 97.5])
    return {"n": int(len(recs)), "eR_median": float(np.median(eR)), "eR_mean": float(eR.mean()), "et_median": float(np.median(et)),
            "sr5": float(np.mean(eR <= 5)), "sr10": float(np.mean(eR <= 10)), "sr10_ci": (float(lo), float(hi)), "sr30": float(np.mean(eR <= 30)),
            "eRsym_median": float(np.median(eRs)), "sr10_sym": float(np.mean(eRs <= 10)), "sr10_sym_ci": (float(lo_s), float(hi_s)),
            "sr30_sym": float(np.mean(eRs <= 30)), "sr5_sym": float(np.mean(eRs <= 5)),
            "add01": float(add.mean()), "ar_mssd": float(ar_s.mean()), "ar_mspd": float(ar_p.mean()),
            "t_median": float(np.nanmedian(tt)) if np.isfinite(tt).any() else float("nan")}


def oracle_summary(recs: list[dict]) -> dict:
    """Best candidate among those retained by the beam (oracle selector): SR@10 / SR@30."""
    best = []
    for r in recs:
        c = r.get("candidates")
        if c:
            best.append(min(float(x["e_R_deg"]) for x in c + [{"e_R_deg": r["e_R_deg"]}]))
    if not best:
        return {}
    b = np.array(best)
    return {"n": int(len(b)), "sr10": float(np.mean(b <= 10)), "sr30": float(np.mean(b <= 30)), "eR_median": float(np.median(b))}


def pct(x: float) -> str:
    return f"{100*x:.0f}"


def f1(x: float, nd: int = 1) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def row(name: str, binlab: str, s: dict) -> str:
    if not s:
        return ""
    return (f"{name} & {binlab} & {s['n']} & {f1(s['eR_median'])} & {pct(s['sr10'])} & {f1(s['eRsym_median'])} & "
            f"{pct(s['sr10_sym'])} [{pct(s['sr10_sym_ci'][0])},{pct(s['sr10_sym_ci'][1])}] & "
            f"{pct(s['add01'])} & {s['ar_mssd']:.2f} & {s['ar_mspd']:.2f} & {ftime(s['t_median'])} \\\\")


def bop_table(all_ds: dict[str, dict[str, list[dict]]]) -> str:
    lines = ["% Auto-generated by tools/export_bop_results.py", "\\begin{table*}[!t]", "\\centering",
             "\\caption{Full 6-DoF object pose on real BOP frames (LM-O, LM, T-LESS; eight objects per dataset, "
             "frames sampled with a fixed seed; the object mask is given, no pose prior). "
             "\\emph{Initialisation-free} methods receive only the mask, depth and RGB crop. \\emph{Refiners} start from the same "
             "perturbed poses $T_0$ in four rotation-error bins (deg) with distance scale in $[0.7,1.4]$ and lateral offset $\\le5\\%$. "
             "$e_R$ / SR@$10^\\circ$: median rotation error (deg) and success rate with the raw rotation error; $e_R^{\\mathrm{sym}}$ / SR@$10^\\circ$ sym.: "
             "the same with the rotation error taken as the minimum over the object's symmetry transforms listed in \\texttt{models\\_info.json} "
             "(identity for asymmetric objects; continuous symmetries sampled every $5^\\circ$), with bootstrap 95\\% CI; ADD(-S)$<0.1d$ recall; "
             "BOP average recalls AR$_{\\mathrm{MSSD}}$ and AR$_{\\mathrm{MSPD}}$; median seconds per frame (RTX~5090, shared GPU; "
             "SCFlow2 on an RTX~2080\\,Ti under its CUDA~11.7 toolchain, so its time is not comparable to the other rows). "
             "The oracle row uses the raw error (the retained candidates store only $e_R$).}",
             "\\label{tab:bop}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{tabular}{@{}llccccccccc@{}}", "\\toprule",
             "Method & $T_0$ bin & $N$ & med.\\ $e_R$ & SR@10$^\\circ$ & med.\\ $e_R^{\\mathrm{sym}}$ & SR@10$^\\circ$ sym.\\ [CI] & ADD(-S)$<0.1d$ & AR$_{\\mathrm{MSSD}}$ & AR$_{\\mathrm{MSPD}}$ & s/frame \\\\"]
    any_row = False
    for ds, disp in DATASETS:
        g = all_ds.get(ds, {})
        block = []
        s = summarize(g.get("ours_full", []))
        if s:
            block.append(row("\\textbf{Proposed (stages I--IV)}", "none", s))
        s = summarize(g.get("ours_full_unbal", []))
        if s:
            block.append(row("\\quad synthetic loss scaling, no lateral polish", "none", s))
        s = summarize(g.get("ours_full_occ", []))
        if s:
            block.append(row("\\quad occlusion-aware silhouette term (margin $r/4$)", "none", s))
        o = oracle_summary(g.get("ours_full", []))
        if o:
            block.append(f"\\quad oracle choice among the retained candidates (raw $e_R$) & none & {o['n']} & {f1(o['eR_median'])} & {pct(o['sr10'])} & -- & -- & -- & -- & -- & -- \\\\")
        s = summarize(g.get("ours_wapr", []))
        if s:
            block.append(row("Proposed stages I--III $\\rightarrow$ WAPR refine", "none", s))
        s = summarize(g.get("wapr_full", []))
        if s:
            block.append(row("WAPR (12 hypotheses + refine)", "none", s))
        for b in BINS:
            for key, name in (("ours_refine", "Proposed stages II--IV"), ("wapr_refine", "WAPR refine"), ("scflow2", "SCFlow2"), ("diffdope", "Diff-DOPE"), ("sdfr", "SDFR")):
                recs = [r for r in g.get(key, []) if str(r.get("bin")) == b]
                s = summarize(recs)
                if s:
                    block.append(row(name, b, s))
        if block:
            any_row = True
            lines += ["\\midrule", f"\\multicolumn{{11}}{{@{{}}l}}{{\\textit{{{disp}}}}} \\\\"] + block
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return ("\n".join(lines) + "\n") if any_row else ""


def fit_width(tex: str) -> str:
    """Scale a tabular down to the text width when it is too wide (adjustbox; never scales up)."""
    return tex.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}", 1)\
              .replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}", 1)


def ftime(x) -> str:
    """median seconds per frame; sub-second values are shown as <1 instead of 0"""
    try:
        if x != x:
            return "--"
        return "$<$1" if 0 <= x < 0.5 else f"{x:.0f}"
    except Exception:
        return "--"


def bop_table_compact(all_ds: dict[str, dict[str, list[dict]]]) -> str:
    """Main-text version: one row per method and dataset; refiners report SR@10 per T0 bin in one cell."""
    lines = ["% Auto-generated by tools/export_bop_results.py", "\\begin{table}[!t]", "\\centering",
             "\\caption{Full 6-DoF object pose on real BOP frames (LM-O, LM, T-LESS; 8 objects per dataset, 104 frames). "
             "Initialisation-free methods receive only mask, depth and RGB; refiners start from the same perturbed poses $T_0$ in four "
             "rotation-error bins ($15$--$30$, $30$--$60$, $60$--$90$, $90$--$180^\\circ$). "
             "SR@$10^\\circ$ in \\% with the rotation error taken as the minimum over the object's symmetry transforms of \\texttt{models\\_info.json} "
             "(identity for asymmetric objects; raw value in brackets for the initialisation-free rows); refiners started from $T_0$ "
             "give it per bin ($15$--$30$ / $30$--$60$ / $60$--$90$ / $90$--$180^\\circ$); "
             "ADD(-S)$<0.1d$ recall over all frames (refiners: all bins); "
             "AR$_{\\mathrm{MSSD}}$: BOP average recall; s: median seconds per frame (shared RTX~5090; SCFlow2 on an RTX~2080\\,Ti, "
             "not comparable). Supplementary Table~S7 adds medians, confidence intervals, MSPD and the ablation and oracle rows.}",
             "\\label{tab:bop}", "\\normalsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
             "Method & SR@10$^\\circ$ sym. & ADD(-S) & AR$_{\\mathrm{MSSD}}$ & s \\\\"]
    any_row = False
    for ds, disp in DATASETS:
        g = all_ds.get(ds, {})
        block = []

        def one(name, key):
            s = summarize(g.get(key, []))
            if s:
                block.append(f"{name} & {pct(s['sr10_sym'])} ({pct(s['sr10'])}) & {pct(s['add01'])} & {s['ar_mssd']:.2f} & {ftime(s['t_median'])} \\\\")

        def bins(name, key):
            cells, allr = [], []
            for b in BINS:
                recs = [r for r in g.get(key, []) if str(r.get("bin")) == b]
                s = summarize(recs)
                cells.append(pct(s["sr10_sym"]) if s else "--")
                allr += recs
            s = summarize(allr)
            if s:
                block.append(f"{name} & {' / '.join(cells)} & {pct(s['add01'])} & {s['ar_mssd']:.2f} & {ftime(s['t_median'])} \\\\")

        one("\\textbf{Proposed, no $T_0$}", "ours_full")
        one("Proposed I--III $\\rightarrow$ WAPR", "ours_wapr")
        one("WAPR 12 hyp., no $T_0$", "wapr_full")
        bins("Proposed II--IV, $T_0$", "ours_refine")
        bins("WAPR refine, $T_0$", "wapr_refine")
        bins("SCFlow2, $T_0$", "scflow2")
        bins("Diff-DOPE, $T_0$", "diffdope")
        bins("SDFR, $T_0$", "sdfr")
        if block:
            any_row = True
            lines += ["\\midrule", f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{disp}}}}} \\\\"] + block
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return ("\n".join(lines) + "\n") if any_row else ""


def fminit_table(all_ds: dict[str, dict[str, list[dict]]]) -> str:
    lines = ["% Auto-generated by tools/export_bop_results.py", "\\begin{table}[!t]", "\\centering",
             "\\caption{Foundation-model initialisation. VGGT (CVPR~2025), $\\pi^3$ (ICLR~2026) and Wid3R (ECCV~2026) receive the real RGB crop and "
             "12 rendered templates of the mesh with known poses; the predicted template cameras are aligned to the known ones by a similarity "
             "transform, which yields an object pose for the real frame (no mask, no depth). CAM3R (ECCV~2026), whose public release is its "
             "two-view network, is run on the 12 (real crop, template) pairs; the pair with the highest predicted confidence gives the pose, its "
             "scale fixed by the template geometry. The pose is then refined by stages II--IV of the proposed method. Same frames as "
             "Table~\\ref{tab:bop} of the main paper; rotation metrics raw / symmetry-aware (sym.). Diagnostics (ground truth used only "
             "here): the median relative residual of the template-camera alignment is \\BopWidFitRes{} for Wid3R against \\BopPiFitRes{} "
             "for $\\pi^3$, and the best of the 12 CAM3R pairs has a median rotation error of \\BopCamBestPair$^\\circ$.}",
             "\\label{tab:bop_fminit}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}llccccccc@{}}", "\\toprule",
             "Dataset & Method & $N$ & med.\\ $e_R$ & SR@10$^\\circ$ & SR@10$^\\circ$ sym.\\ & SR@30$^\\circ$ sym.\\ & ADD(-S)$<0.1d$ & s/frame \\\\", "\\midrule"]
    any_row = False
    for ds, disp in DATASETS:
        g = all_ds.get(ds, {})
        block = []
        for key, name in (("vggt_init", "VGGT init only"), ("vggt_ours", "VGGT $\\rightarrow$ stages II--IV"),
                          ("pi3_init", "$\\pi^3$ init only"), ("pi3_ours", "$\\pi^3$ $\\rightarrow$ stages II--IV"),
                          ("wid3r_init", "Wid3R init only"), ("wid3r_ours", "Wid3R $\\rightarrow$ stages II--IV"),
                          ("cam3r_init", "CAM3R init only"), ("cam3r_ours", "CAM3R $\\rightarrow$ stages II--IV"),
                          ("ours_full", "Proposed (stages I--IV, no initialisation)")):
            s = summarize(g.get(key, []))
            if s:
                block.append(f"{disp if not block else ''} & {name} & {s['n']} & {f1(s['eR_median'])} & {pct(s['sr10'])} & {pct(s['sr10_sym'])} & {pct(s['sr30_sym'])} & {pct(s['add01'])} & {ftime(s['t_median'])} \\\\")
        if block:
            any_row = True
            lines += block + ["\\midrule"]
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return ("\n".join(lines) + "\n") if any_row else ""


def stub(label: str, cap: str) -> str:
    return f"\\begin{{table}}[t]\\centering\\small\\caption{{{cap}}}\\label{{{label}}}\\begin{{tabular}}{{l}}\\toprule PENDING \\\\ \\bottomrule\\end{{tabular}}\\end{{table}}\n"


def shell_range_stats(all_ds) -> dict:
    """Camera distance / mesh radius of every BOP frame (ground truth, evaluation only) against the default shell range
    [0.4, 6.0] r and the fixed 0.5r-4r list of the radius ablation, and the share of frames whose estimated camera centre
    (proposed method, stages I-IV) lies within 15% of the true distance."""
    ratios, rel = [], []
    for ds, _ in DATASETS:
        dist = {}
        for pj in sorted((RES / ds).glob("*/protocol.json")):
            j = json.loads(pj.read_text(encoding="utf-8"))
            R = np.array(j["gt_R"], dtype=np.float64).reshape(3, 3)
            t = np.array(j["gt_t"], dtype=np.float64).reshape(3)
            c = np.array(j.get("mesh_centre_obj", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            dist[pj.parent.name] = float(np.linalg.norm(R @ c + t))
            ratios.append(dist[pj.parent.name] / float(j["mesh_r"]))
        for r in all_ds.get(ds, {}).get("ours_full", []):
            d = dist.get(str(r.get("tag")))
            if d and r.get("e_t_mm") is not None:
                rel.append(float(r["e_t_mm"]) / d)
    if not ratios:
        return {}
    v, e = np.array(ratios), np.array(rel)
    return {"n": len(v), "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()),
            "out_default": float(np.mean((v > 6.0) | (v < 0.4))), "out_fixed": float(np.mean((v > 4.0) | (v < 0.5))),
            "within15": float(np.mean(e < 0.15)) if len(e) else float("nan"), "n_est": len(e)}


def macros(all_ds) -> str:
    L = ["% Auto-generated by tools/export_bop_results.py"]
    sr = shell_range_stats(all_ds)
    if sr:
        L += [f"\\newcommand{{\\BopDrMin}}{{{sr['min']:.1f}}}", f"\\newcommand{{\\BopDrMed}}{{{sr['median']:.1f}}}",
              f"\\newcommand{{\\BopDrMax}}{{{sr['max']:.1f}}}", f"\\newcommand{{\\BopDrN}}{{{sr['n']}}}",
              f"\\newcommand{{\\BopDrOutDefault}}{{{pct(sr['out_default'])}}}", f"\\newcommand{{\\BopDrOutFixed}}{{{pct(sr['out_fixed'])}}}",
              f"\\newcommand{{\\BopDrWithin}}{{{pct(sr['within15'])}}}"]
    for ds, _ in DATASETS:
        g = all_ds.get(ds, {})
        key = {"lmo": "LMO", "lm": "LM", "tless": "TLESS"}[ds]
        for mk, suffix in (("ours_full", "Ours"), ("wapr_full", "WaprFull"), ("vggt_init", "Vggt"), ("vggt_ours", "VggtOurs"),
                           ("pi3_init", "Pi"), ("pi3_ours", "PiOurs"), ("wid3r_init", "Wid"), ("wid3r_ours", "WidOurs"),
                           ("cam3r_init", "Cam"), ("cam3r_ours", "CamOurs"), ("ours_full_occ", "OursOcc"), ("scflow2", "ScflowAll"),
                           ("ours_full_unbal", "OursUnbal"), ("ours_wapr", "OursWapr"), ("diffdope", "DiffDopeAll"), ("wapr_refine", "WaprRefAll"), ("ours_refine", "OursRefAll"),
                           ("sdfr", "SdfrAll")):
            s = summarize(g.get(mk, []))
            if s:
                L += [f"\\newcommand{{\\Bop{key}{suffix}SRtenSym}}{{{pct(s['sr10_sym'])}}}", f"\\newcommand{{\\Bop{key}{suffix}SRthirtySym}}{{{pct(s['sr30_sym'])}}}",
                      f"\\newcommand{{\\Bop{key}{suffix}ERsym}}{{{f1(s['eRsym_median'])}}}",
                      f"\\newcommand{{\\Bop{key}{suffix}SRten}}{{{pct(s['sr10'])}}}", f"\\newcommand{{\\Bop{key}{suffix}SRthirty}}{{{pct(s['sr30'])}}}",
                      f"\\newcommand{{\\Bop{key}{suffix}ER}}{{{f1(s['eR_median'])}}}", f"\\newcommand{{\\Bop{key}{suffix}ET}}{{{f1(s['et_median'], 0)}}}",
                      f"\\newcommand{{\\Bop{key}{suffix}ADD}}{{{pct(s['add01'])}}}", f"\\newcommand{{\\Bop{key}{suffix}N}}{{{s['n']}}}",
                      f"\\newcommand{{\\Bop{key}{suffix}ARmssd}}{{{s['ar_mssd']:.2f}}}", f"\\newcommand{{\\Bop{key}{suffix}Time}}{{{ftime(s['t_median'])}}}"]
        for b in BINS:
            bk = BIN_KEY[b]
            for mk, suffix in (("ours_refine", "OursRef"), ("wapr_refine", "WaprRef"), ("diffdope", "DiffDope"), ("scflow2", "Scflow"), ("sdfr", "Sdfr")):
                s = summarize([r for r in g.get(mk, []) if str(r.get("bin")) == b])
                if s:
                    L += [f"\\newcommand{{\\Bop{key}{suffix}{bk}SRten}}{{{pct(s['sr10'])}}}", f"\\newcommand{{\\Bop{key}{suffix}{bk}ADD}}{{{pct(s['add01'])}}}",
                          f"\\newcommand{{\\Bop{key}{suffix}{bk}ER}}{{{f1(s['eR_median'])}}}", f"\\newcommand{{\\Bop{key}{suffix}{bk}SRtenSym}}{{{pct(s['sr10_sym'])}}}"]
        o = oracle_summary(g.get("ours_full", []))
        if o:
            L += [f"\\newcommand{{\\Bop{key}OursOracleSRten}}{{{pct(o['sr10'])}}}", f"\\newcommand{{\\Bop{key}OursOracleSRthirty}}{{{pct(o['sr30'])}}}",
                  f"\\newcommand{{\\Bop{key}OursOracleN}}{{{o['n']}}}"]
    gain = loss = n_occ = 0
    for ds, _ in DATASETS:
        g = all_ds.get(ds, {})
        base = {r["tag"]: r for r in g.get("ours_full", [])}
        for r in g.get("ours_full_occ", []):
            b0 = base.get(r["tag"])
            if b0 is None:
                continue
            n_occ += 1
            gain += int(r["e_R_sym_deg"] <= 10 < b0["e_R_sym_deg"])
            loss += int(b0["e_R_sym_deg"] <= 10 < r["e_R_sym_deg"])
    if n_occ:
        m_ = gain + loss
        p_occ = min(1.0, 2 * sum(math.comb(m_, i) for i in range(min(gain, loss) + 1)) / 2 ** m_) if m_ else 1.0
        L += [f"\\newcommand{{\\BopOccN}}{{{n_occ}}}", f"\\newcommand{{\\BopOccGain}}{{{gain}}}", f"\\newcommand{{\\BopOccLoss}}{{{loss}}}",
              f"\\newcommand{{\\BopOccP}}{{{p_occ:.2f}}}"]
    fit_w = [float(r["fit_residual_rel"]) for ds, _ in DATASETS for r in read_jsonl(RES / ds / "baseline_wid3r_init.jsonl") if "fit_residual_rel" in r]
    fit_p = [float(r["fit_residual_rel"]) for ds, _ in DATASETS for r in read_jsonl(RES / ds / "baseline_pi3_init.jsonl") if "fit_residual_rel" in r]
    best_c = [float(r["e_R_best_pair_deg_diag"]) for ds, _ in DATASETS for r in read_jsonl(RES / ds / "baseline_cam3r_init.jsonl") if "e_R_best_pair_deg_diag" in r]
    if fit_w and fit_p:
        L += [f"\\newcommand{{\\BopWidFitRes}}{{{np.median(fit_w):.2f}}}", f"\\newcommand{{\\BopPiFitRes}}{{{np.median(fit_p):.2f}}}"]
    if best_c:
        L.append(f"\\newcommand{{\\BopCamBestPair}}{{{np.median(best_c):.0f}}}")
    defined = {ln.split("{")[1].split("}")[0] for ln in L if ln.startswith("\\newcommand")}
    expected = []
    for key in ("LMO", "LM", "TLESS"):
        for suffix in ("Ours", "WaprFull", "Vggt", "VggtOurs", "Pi", "PiOurs", "Wid", "WidOurs", "Cam", "CamOurs", "OursOcc", "OursUnbal", "OursWapr",
                       "DiffDopeAll", "WaprRefAll", "OursRefAll", "ScflowAll", "SdfrAll"):
            expected += [f"Bop{key}{suffix}{s}" for s in ("SRten", "SRthirty", "ER", "ET", "ADD", "N", "ARmssd", "Time", "SRtenSym", "SRthirtySym", "ERsym")]
        expected += [f"Bop{key}OursOracleSRten", f"Bop{key}OursOracleSRthirty", f"Bop{key}OursOracleN"]
        for b in BINS:
            bk = BIN_KEY[b]
            for suffix in ("OursRef", "WaprRef", "DiffDope", "Scflow", "Sdfr"):
                expected += [f"Bop{key}{suffix}{bk}SRten", f"Bop{key}{suffix}{bk}ADD", f"Bop{key}{suffix}{bk}ER", f"Bop{key}{suffix}{bk}SRtenSym"]
    expected += ["BopOccN", "BopOccGain", "BopOccLoss", "BopOccP", "BopWidFitRes", "BopPiFitRes", "BopCamBestPair"]
    L.append("% placeholders (no data yet)")
    for name in expected:
        if ("\\" + name) not in defined:
            L.append(f"\\providecommand{{\\{name}}}{{\\textbf{{??}}}}")
    return "\n".join(L) + "\n"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    all_ds = {ds: load_all(ds) for ds, _ in DATASETS}
    t = bop_table_compact(all_ds)
    (TABLES / "tab_bop.tex").write_text(fit_width(t) if t else stub("tab:bop", "BOP full 6-DoF evaluation: results pending."), encoding="utf-8")
    tf = bop_table(all_ds).replace("\\label{tab:bop}", "\\label{tab:bop_full}").replace("\\begin{table*}[!t]", "\\begin{table}[!t]").replace("\\end{table*}", "\\end{table}")
    full = fit_width(tf).replace("{max width=\\linewidth}", "{max width=\\linewidth,max totalheight=0.74\\textheight}", 1) if tf else ""
    (TABLES / "tab_bop_full.tex").write_text(full if tf else stub("tab:bop_full", "BOP full 6-DoF evaluation: results pending."), encoding="utf-8")
    t2 = fminit_table(all_ds)
    (TABLES / "tab_bop_fminit.tex").write_text(fit_width(t2) if t2 else stub("tab:bop_fminit", "Foundation-model initialisation: results pending."), encoding="utf-8")
    (MAN / "bop_macros.tex").write_text(macros(all_ds), encoding="utf-8")
    summary = {ds: {k: summarize(v) for k, v in g.items() if v} for ds, g in all_ds.items()}
    for ds, g in all_ds.items():
        for b in BINS:
            for k in ("ours_refine", "wapr_refine", "diffdope"):
                s = summarize([r for r in g.get(k, []) if str(r.get("bin")) == b])
                if s:
                    summary[ds][f"{k}@{b}"] = s
    (ROOT / "benchmark_results").mkdir(exist_ok=True)
    (ROOT / "benchmark_results" / "bop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for ds, g in summary.items():
        for k, s in g.items():
            print(f"{ds:6s} {k:22s} n={s['n']:3d} medER={s['eR_median']:6.1f} SR10={100*s['sr10']:5.1f}% | sym: medER={s['eRsym_median']:6.1f} SR10={100*s['sr10_sym']:5.1f}% | ADD={100*s['add01']:5.1f}% ARmssd={s['ar_mssd']:.2f} t={s['t_median']:.0f}s")
    print(f"[bop] symmetry check: {SYM_CHECK['n']} records, max |e_R(recomputed) - e_R(stored)| = {SYM_CHECK['max_dev']:.2e} deg")
    print("[bop] tables written:", "tab_bop.tex" if t else "tab_bop.tex (stub)", "|", "tab_bop_fminit.tex" if t2 else "tab_bop_fminit.tex (stub)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
