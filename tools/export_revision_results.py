#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export revision results (strict no-GT, metric-depth benchmark) to the manuscript:
LaTeX tables, number macros, and figures. Runs on partial results (missing runs are skipped).

Outputs (manuscript/computational_visual_media/):
  tables/tab_results_bunny100gt.tex, tables/tab_results_dragon100gt.tex   main tables (N=100)
  tables/tab_new_meshes.tex           armadillo / happy / lucy (N=50)
  tables/tab_cross_object.tex         five meshes, key methods
  tables/tab_ablation_da.tex          distance alignment control (matched views)
  tables/tab_ablation_radius.tex      radius schedule, equal budget (matched views)
  tables/tab_runtime.tex              per-stage runtimes (mean / median / P95)
  tables/tab_transfer.tex             bunny <-> dragon paired differences with bootstrap CIs
  results_macros.tex                  numbers used in abstract / text
  figures/exp/fig_binned_sr.pdf       SR@10 vs GT elevation and distance ratio bins
  figures/exp/fig_runtime.pdf         runtime bars
  benchmark_results/revision_summary.json

Usage:  python tools/export_revision_results.py
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import zlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_geom import full_pose_metrics_row  # noqa: E402

RUNS = ROOT / "benchmark_runs"
MAN = ROOT / "manuscript" / "computational_visual_media"
TABLES = MAN / "tables"
FIGS = MAN / "figures" / "exp"
RNG = np.random.default_rng(12291)
LATE_METHODS = {"foundationpose_full", "foundationpose_refine", "foundationpose_sanity", "ours_v10_stage1"}
LATE_SEED = 12292
# tables of the supplementary material (placed with [!htbp]); the others are main-text tables
SUPP_TABLES = {"tab_results_full.tex", "tab_new_meshes.tex", "tab_runtime.tex", "tab_transfer.tex", "tab_ablation_radius.tex", "tab_corr.tex"}

MESHES = {  # dataset key -> (display, N for main table, radius fallback)
    "bunny": ("Bunny", 100, 0.116616),
    "dragon": ("Dragon", 100, 119.393555),
    "armadillo": ("Armadillo", 50, 100.2637),
    "happy": ("Happy Buddha", 50, 0.1110),
    "lucy": ("Lucy", 50, 940.0511),
}
GROUPS = [
    ("Proposed", [("ours_v10", "Proposed (progressive cue fusion)")]),
    ("Internal ablations", [
        ("ours_v10_geom", "Geom-only selection (no RGB)"),
        ("ours_v10_noda", "w/o distance alignment (w/o-DA)"),
        ("greedy_bt_local", "Greedy-BT + local (width-one walk)"),
        ("rand_topk_only", "Rand-TopK only (best proposal)"),
        ("ours_v10_stage1", "Stage I only (lowest-loss proposal, no beam)"),
    ]),
    ("Recent methods on the same observations", [
        ("wapr_full", "WAPR, 12 hypotheses"),
        ("wapr_refine", "WAPR refine$^{\\ast}$"),
        ("foundationpose_full", "FoundationPose (registration)"),
        ("foundationpose_refine", "FoundationPose refine$^{\\ast}$"),
        ("diffdope", "Diff-DOPE$^{\\ast}$"),
        ("sdfr", "SDFR$^{\\ast}$"),
    ]),
    ("Fair DR baselines (global search + Adam)", [
        ("grid_adam", "Grid+Adam"),
        ("grid_adam_top16", "Grid+Adam-Top16"),
        ("rand_restart_adam", "RandRestart+Adam"),
    ]),
    ("Weak DR baselines (single start)", [
        ("adam_dr", "Adam-DR"),
        ("hybrid_dr", "Hybrid-DR"),
    ]),
]
NAME = {m: n for _, lst in GROUPS for m, n in lst}
# GROUPS fixes the order in which the rows are computed (the bootstrap stream depends on it and must not change);
# DISPLAY fixes the order and the grouping in which they are printed. Greedy-BT and Rand-TopK were run with the
# configuration of the DR baselines (s_img = 1, fixed 36 shells) and are shown with them; the internal ablations
# share the proposals, the loss and the seeds of the proposed run.
DISPLAY = [
    ("Proposed", ["ours_v10"]),
    ("Internal ablations", ["ours_v10_geom", "ours_v10_noda", "ours_v10_stage1"]),
    ("Recent methods on the same observations", ["wapr_full", "wapr_refine", "foundationpose_full", "foundationpose_refine", "diffdope", "sdfr"]),
    ("Proposal-level and DR baselines ($s_{\\mathrm{img}}{=}1$, fixed shells)", ["rand_topk_only", "greedy_bt_local", "grid_adam", "grid_adam_top16", "rand_restart_adam"]),
    ("Weak DR baselines (single start)", ["adam_dr", "hybrid_dr"]),
]
assert sorted(m for _, lst in DISPLAY for m in lst) == sorted(NAME), "DISPLAY and GROUPS must list the same methods"


def emit(row_of: dict, n_cols: int, first_header: bool = False) -> list[str]:
    """Rows computed in GROUPS order, printed in DISPLAY order with one italic header per non-empty group."""
    out = []
    for gname, lst in DISPLAY:
        glines = [row_of[m] for m in lst if m in row_of]
        if glines:
            if gname != "Proposed" or first_header:
                out.append(f"\\multicolumn{{{n_cols}}}{{@{{}}l}}{{\\textit{{{gname}}}}} \\\\")
            out += glines
    return out
NAME.update({"ours_v10_fixedmr": "Fixed multi-radius (36 shells)", "ours_v10_1shell": "Single shell ($2r$)"})
SHORT = {"ours_v10": "Proposed", "ours_v10_noda": "w/o-DA", "ours_v10_geom": "Geom-only", "greedy_bt_local": "Greedy-BT",
         "rand_topk_only": "Rand-TopK", "grid_adam": "Grid+Adam", "grid_adam_top16": "Grid+Adam-T16",
         "rand_restart_adam": "RandRestart", "adam_dr": "Adam-DR", "hybrid_dr": "Hybrid-DR",
         "ours_v10_fixedmr": "Fixed-MR", "ours_v10_1shell": "1-shell",
         "wapr_full": "WAPR (12 hyp.)", "wapr_refine": "WAPR refine", "diffdope": "Diff-DOPE", "sdfr": "SDFR",
         "foundationpose_full": "FoundationPose", "foundationpose_refine": "FoundationPose refine",
         "ours_v10_fast": "Proposed, 1/8 budget", "ours_v10_stage1": "Stage I only"}
# near-duplicate baselines and FoundationPose's refiner, shown only in the complete table of the supplement
MAIN_SKIP = {"grid_adam_top16", "hybrid_dr", "foundationpose_refine"}
# compact names for the 10 pt main table
TAB1_NAME = {"ours_v10": "Proposed", "ours_v10_geom": "Geom-only (no RGB)", "ours_v10_noda": "w/o-DA", "greedy_bt_local": "Greedy-BT + local",
             "rand_topk_only": "Rand-TopK (best proposal)", "ours_v10_stage1": "Stage I only (no beam)",
             "wapr_full": "WAPR, 12 hyp.", "wapr_refine": "WAPR refine$^{\\ast}$",
             "foundationpose_full": "FoundationPose, registration",
             "diffdope": "Diff-DOPE$^{\\ast}$", "sdfr": "SDFR$^{\\ast}$", "grid_adam": "Grid+Adam", "grid_adam_top16": "Grid+Adam-Top16",
             "rand_restart_adam": "RandRestart+Adam", "adam_dr": "Adam-DR", "hybrid_dr": "Hybrid-DR"}


# ----------------------------------------------------------------------------- loading
def load_rows(ds: str, n: int, method: str) -> list[dict]:
    p = RUNS / f"{ds}_{n}gt" / method / "pose_metrics_summary.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    out = []
    for r in rows:
        try:
            if r.get("cam_center_err") in (None, "", "nan"):
                base = {k: float(r[k]) for k in ("gt_dist", "gt_elev_deg", "gt_azim_deg", "est_dist", "est_elev_deg", "est_azim_deg")}
                r = {**r, **full_pose_metrics_row(base)}
            if method in LATE_METHODS:
                r = {**r, "_late": True}
            out.append(r)
        except Exception:
            continue
    return out


def mesh_radius(ds: str, n: int) -> float:
    log = RUNS / f"{ds}_{n}gt" / "ours_v10" / "run.log"
    if log.exists():
        try:
            txt = log.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"\br=([0-9.]+)\s+radii_mults", txt)
            if m:
                return float(m.group(1))
        except Exception:
            pass
    return MESHES[ds][2]


def fl(r: dict, k: str) -> float:
    return float(r[k])


def boot_ci(vals: np.ndarray, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(vals), size=(n_boot, len(vals)))
    bs = stat(vals[idx], axis=1)
    return (float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2)))


def summarize(rows: list[dict], r_mesh: float) -> dict:
    global RNG
    if rows and rows[0].get("_late"):
        # methods added after the tables were published draw their bootstrap from a separate stream, so that the
        # confidence intervals of every other row stay exactly as reported
        saved, RNG = RNG, np.random.default_rng(LATE_SEED)
        try:
            return summarize([{k: v for k, v in r.items() if k != "_late"} for r in rows], r_mesh)
        finally:
            RNG = saved
    if not rows:
        return {}
    eR = np.array([fl(r, "rot_err_deg") for r in rows])
    et = np.array([fl(r, "cam_center_err") for r in rows]) / r_mesh
    dd = np.array([fl(r, "dist_err") for r in rows])
    dgt = np.array([fl(r, "gt_dist") for r in rows])
    shell_wrong = np.mean(dd / np.maximum(dgt, 1e-9) > 0.25)
    out = {
        "n": len(rows),
        "eR_mean": float(eR.mean()), "eR_ci": boot_ci(eR), "eR_median": float(np.median(eR)), "eR_std": float(eR.std(ddof=1)) if len(eR) > 1 else 0.0,
        "sr5": float(np.mean(eR <= 5)), "sr10": float(np.mean(eR <= 10)), "sr15": float(np.mean(eR <= 15)),
        "et_r_mean": float(et.mean()), "et_r_median": float(np.median(et)),
        "dd_r_mean": float((dd / r_mesh).mean()), "dd_r_median": float(np.median(dd / r_mesh)),
        "wrong_shell": float(shell_wrong),
    }
    if "t_total_s" in rows[0] and str(rows[0].get("t_total_s", "")).strip() not in ("", "nan"):
        tt = np.array([fl(r, "t_total_s") for r in rows if str(r.get("t_total_s", "")).strip() not in ("", "nan")])
        if len(tt):
            out.update({"t_mean": float(tt.mean()), "t_median": float(np.median(tt)), "t_p95": float(np.quantile(tt, 0.95))})
            for k in ("t_global_s", "t_dist_align_s", "t_local_s", "t_select_s", "t_refine_s"):
                v = np.array([fl(r, k) for r in rows if str(r.get(k, "")).strip() not in ("", "nan")])
                if len(v):
                    out[k + "_mean"] = float(v.mean())
    return out


def f1(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def pct(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100*x:.0f}\\%"


# ----------------------------------------------------------------------------- tables
def main_table(ds: str) -> str:
    disp, n, _ = MESHES[ds]
    r_mesh = mesh_radius(ds, n)
    lines = [
        "% Auto-generated by tools/export_revision_results.py",
        "\\begin{table}[!t]", "\\centering",
        f"\\caption{{Pose errors on the Stanford {disp} ($N{{=}}{n}$ views; strict no-ground-truth protocol, metric depth). "
        "$e_R$: geodesic rotation error (deg) with bootstrap 95\\% CI of the mean; SR@$\\tau$: fraction with $e_R\\le\\tau^\\circ$; "
        "$e_t/r$ and $|\\Delta d|/r$: camera-centre and distance errors in units of the mesh bounding radius (medians); "
        "wrong shell: fraction with $|\\Delta d|>0.25\\,d_{\\mathrm{gt}}$; time: median seconds per view on one RTX~5090 (shared). "
        "Rows marked $\\dagger$ use the first $N'$ views of the same set.}",
        f"\\label{{tab:{ds}100gt_main}}" if ds in ("bunny", "dragon") else f"\\label{{tab:{ds}_main}}",
        "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
        "Method & $e_R$ [95\\% CI] & med.\\ $e_R$ & SR@5$^\\circ$ & SR@10$^\\circ$ & $e_t/r$ & $|\\Delta d|/r$ & wrong shell \\\\",
        "\\midrule",
    ]
    row_of = {}
    for gname, lst in GROUPS:
        for m, mname in lst:
            rows = load_rows(ds, n, m)
            tag = ""
            if not rows and m in ("ours_v10_noda",):
                for n2 in (50, 30):
                    rows = load_rows(ds, n2, m)
                    if rows:
                        tag = f"$^\\dagger$($N'{{=}}{n2}$)"
                        break
            if not rows:
                continue
            s = summarize(rows, r_mesh)
            bold = m == "ours_v10"
            name = ("\\textbf{" + mname + "}" if bold else mname) + tag
            row_of[m] = (
                f"{name} & {f1(s['eR_mean'])} [{f1(s['eR_ci'][0])}, {f1(s['eR_ci'][1])}] & {f1(s['eR_median'])} & "
                f"{pct(s['sr5'])} & {pct(s['sr10'])} & {f1(s['et_r_median'],3)} & {f1(s['dd_r_median'],3)} & {pct(s['wrong_shell'])} \\\\"
            )
    lines += emit(row_of, 8, first_header=True)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def _rows_for(ds: str, n: int, m: str) -> tuple[list[dict], str]:
    rows = load_rows(ds, n, m)
    tag = ""
    if not rows or (len(rows) < n and m != "ours_v10"):
        for n2 in (100, 50, 30):
            rows2 = load_rows(ds, n2, m) if n2 != n else rows
            if rows2:
                rows = rows2
                if len(rows) != n:
                    tag = "$^\\dagger$"
                break
    if rows and len(rows) != n and not tag:
        tag = "$^\\dagger$"
    return rows, tag


def main_table_compact() -> str:
    """Main-text table at the journal's 10 pt: mean / median e_R, SR@10, distance errors; CIs and SR@5 are in tab_results_full."""
    lines = [
        "% Auto-generated by tools/export_revision_results.py",
        "\\begin{table}[!tp]", "\\centering",
        "\\caption{Pose errors on the Stanford Bunny and Dragon ($N{=}100$ views each; strict no-ground-truth protocol, metric depth; "
        "every method receives the same rendered observation). $e_R$: geodesic rotation error (deg), mean and median; "
        "SR@$10^\\circ$: fraction with $e_R\\le10^\\circ$; $e_t/r$, $|\\Delta d|/r$: median camera-centre and distance errors in units of the "
        "bounding radius; wrong shell: fraction with $|\\Delta d|>0.25\\,d_{\\mathrm{gt}}$. $^{\\ast}$Started from the fixed initial camera of the "
        "DR baselines. $^\\dagger$First 50 views. Stage~I only: lowest-loss proposal of the proposed run. Bootstrap CIs, SR@$5^\\circ$, FoundationPose's refiner, Grid+Adam-Top16 and Hybrid-DR: Supplementary Table~S9.}",
        "\\label{tab:bunny100gt_main}\\label{tab:dragon100gt_main}",
        "\\normalsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
        "Method & mean $e_R$ & med.\\ $e_R$ & SR@10$^\\circ$ & $e_t/r$ & $|\\Delta d|/r$ & wrong shell \\\\",
    ]
    for ds in ("bunny", "dragon"):
        disp, n, _ = MESHES[ds]
        r_mesh = mesh_radius(ds, n)
        lines += ["\\midrule", f"\\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{disp}}} ($r{{=}}{r_mesh:.3g}$)}} \\\\"]
        row_of = {}
        for gname, lst in GROUPS:
            for m, _ in lst:
                if m in MAIN_SKIP:
                    continue
                rows, tag = _rows_for(ds, n, m)
                if not rows:
                    continue
                sm = summarize(rows, r_mesh)
                name = TAB1_NAME.get(m, m)
                name = ("\\textbf{" + name + "}" if m == "ours_v10" else name) + tag
                row_of[m] = (f"{name} & {f1(sm['eR_mean'])} & {f1(sm['eR_median'])} & {pct(sm['sr10'])} & "
                             f"{f1(sm['et_r_median'],3)} & {f1(sm['dd_r_median'],3)} & {pct(sm['wrong_shell'])} \\\\")
        lines += emit(row_of, 7)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main_table_combined() -> str:
    """Complete Bunny / Dragon table (supplement): adds the bootstrap CI of the mean and SR@5."""
    lines = [
        "% Auto-generated by tools/export_revision_results.py",
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Complete results on the Stanford Bunny and Dragon (Table~\\ref{tab:bunny100gt_main} of the main paper with the bootstrap 95\\% CI of the "
        "mean and SR@$5^\\circ$; $N{=}100$ views each; strict no-ground-truth protocol, metric depth). "
        "$e_R$: geodesic rotation error (deg) with bootstrap 95\\% CI of the mean; SR@$\\tau$: fraction with $e_R\\le\\tau^\\circ$; "
        "$e_t/r$ and $|\\Delta d|/r$: camera-centre and distance errors in units of the mesh bounding radius (medians); "
        "wrong shell: fraction with $|\\Delta d|>0.25\\,d_{\\mathrm{gt}}$. $^{\\ast}$From the fixed initial camera of the DR baselines. "
        "Rows marked $\\dagger$ use the first $N'$ views of the same seeded sequence. Stage~I only: the lowest-loss proposal of the proposed run, "
        "without any later stage.}",
        "\\label{tab:results_full}",
        "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
        "Method & $e_R$ [95\\% CI] & med.\\ $e_R$ & SR@5$^\\circ$ & SR@10$^\\circ$ & $e_t/r$ & $|\\Delta d|/r$ & wrong shell \\\\",
    ]
    for ds in ("bunny", "dragon"):
        disp, n, _ = MESHES[ds]
        r_mesh = mesh_radius(ds, n)
        lines += ["\\midrule", f"\\multicolumn{{8}}{{@{{}}l}}{{\\textbf{{{disp}}} ($r{{=}}{r_mesh:.3g}$)}} \\\\"]
        row_of = {}
        for gname, lst in GROUPS:
            for m, mname in lst:
                rows = load_rows(ds, n, m)
                tag = ""
                if not rows or (len(rows) < n and m != "ours_v10"):
                    for n2 in (100, 50, 30):
                        rows2 = load_rows(ds, n2, m) if n2 != n else rows
                        if rows2:
                            rows = rows2
                            if len(rows) != n:
                                tag = f"$^\\dagger$($N'{{=}}{len(rows)}$)"
                            break
                if not rows:
                    continue
                if len(rows) != n and not tag:
                    tag = f"$^\\dagger$($N'{{=}}{len(rows)}$)"
                sm = summarize(rows, r_mesh)
                bold = m == "ours_v10"
                name = ("\\textbf{" + mname + "}" if bold else mname) + tag
                row_of[m] = (
                    f"{name} & {f1(sm['eR_mean'])} [{f1(sm['eR_ci'][0])}, {f1(sm['eR_ci'][1])}] & {f1(sm['eR_median'])} & "
                    f"{pct(sm['sr5'])} & {pct(sm['sr10'])} & {f1(sm['et_r_median'],3)} & {f1(sm['dd_r_median'],3)} & {pct(sm['wrong_shell'])} \\\\"
                )
        lines += emit(row_of, 8)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def new_meshes_table() -> str:
    dss = ["armadillo", "happy", "lucy"]
    lines = ["% Auto-generated by tools/export_revision_results.py", "\\begin{table}[!t]", "\\centering",
             "\\caption{Additional Stanford meshes (first $N$ views of a seeded sequence of 50; same configuration as Table~\\ref{tab:bunny100gt_main} of the main paper, no per-mesh tuning; "
             "a bracketed superscript gives $N$ where a run uses fewer than 50 views). "
             f"Bounding radii: Armadillo ${mesh_radius('armadillo', 50):.3g}$, Happy Buddha ${mesh_radius('happy', 50):.2g}$, Lucy ${mesh_radius('lucy', 50):.3g}$ units. "
             "Mean $e_R$ (deg) / SR@$10^\\circ$ / median $|\\Delta d|/r$.}",
             "\\label{tab:new_meshes}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}l" + "ccc" * len(dss) + "@{}}", "\\toprule",
             "& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{MESHES[d][0]}}}" for d in dss) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(dss))),
             "Method & " + " & ".join("$e_R$ & SR@10 & $|\\Delta d|/r$" for _ in dss) + " \\\\", "\\midrule"]
    row_of = {}
    for gname, lst in GROUPS:
        for m, mname in lst:
            cells = []
            have = False
            for d in dss:
                rows = load_rows(d, 50, m) or load_rows(d, 30, m)
                if rows:
                    s = summarize(rows, mesh_radius(d, 50))
                    nn = f"$^{{({len(rows)})}}$" if len(rows) != 50 else ""
                    cells.append(f"{f1(s['eR_mean'])}{nn} & {pct(s['sr10'])} & {f1(s['dd_r_median'],3)}")
                    have = True
                else:
                    cells.append("-- & -- & --")
            if have:
                name = "\\textbf{" + SHORT[m] + "}" if m == "ours_v10" else SHORT.get(m, m)
                row_of[m] = f"{name} & " + " & ".join(cells) + " \\\\"
    lines += [row_of[m] for _, lst in DISPLAY for m in lst if m in row_of]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return ("\n".join(lines) + "\n") if row_of else ""


def cross_object_table() -> str:
    methods = ["ours_v10", "ours_v10_geom", "wapr_full", "foundationpose_full", "wapr_refine", "diffdope", "sdfr", "grid_adam", "rand_restart_adam", "adam_dr"]
    dss = list(MESHES.keys())
    lines = ["% Auto-generated by tools/export_revision_results.py", "\\begin{table}[!t]", "\\centering",
             "\\caption{Cross-object summary: one configuration applied to five meshes spanning four orders of magnitude in bounding radius. "
             "Cells: mean $e_R$ (deg) / SR@$10^\\circ$ (\\%) over the $N$ views of Table~\\ref{tab:bunny100gt_main} and Supplementary Table~S1 "
             "(Bunny, Dragon: $N{=}100$; other meshes: $N{=}50$).}",
             "\\label{tab:cross_object}", "\\normalsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}l" + "c" * len(dss) + "@{}}", "\\toprule",
             "Method & " + " & ".join(MESHES[d][0] for d in dss) + " \\\\",
             " & " + " & ".join(f"$r{{=}}{mesh_radius(d, MESHES[d][1]):.3g}$" for d in dss) + " \\\\", "\\midrule"]
    for m in methods:
        cells = []
        have = False
        for d in dss:
            n_main = MESHES[d][1]
            rows = load_rows(d, n_main, m) or load_rows(d, 50, m) or load_rows(d, 30, m)
            n_exp = n_main
            if rows and len(rows) >= 10:
                s = summarize(rows, mesh_radius(d, n_main))
                tag = f" [{len(rows)}]" if len(rows) < n_exp else ""
                cells.append(f"{f1(s['eR_mean'],1)} / {100 * s['sr10']:.0f}{tag}")
                have = True
            else:
                cells.append("--")
        if have:
            name = "\\textbf{Proposed}" if m == "ours_v10" else SHORT.get(m, m)
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def matched(ds: str, m_a: str, m_b: str, n_a: int, n_b: int) -> tuple[list[dict], list[dict]]:
    ra = {r["tag"]: r for r in load_rows(ds, n_a, m_a)}
    rb = {r["tag"]: r for r in load_rows(ds, n_b, m_b)}
    tags = sorted(set(ra) & set(rb))
    return [ra[t] for t in tags], [rb[t] for t in tags]


def da_table() -> str:
    lines = ["% Auto-generated by tools/export_revision_results.py", "\\begin{table}[!t]", "\\centering",
             "\\caption{Distance-alignment control on matched views (identical global search, beam and selector; only stage~II and the hint-dependent terms removed). "
             "Paired bootstrap 95\\% CI of the mean $e_R$ difference (w/o-DA $-$ Proposed).}",
             "\\label{tab:ablation_da}", "\\normalsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}llcccccc@{}}", "\\toprule",
             "Mesh & Variant & $N$ & $e_R$ & SR@10$^\\circ$ & $|\\Delta d|/r$ & wrong shell & $\\Delta e_R$ [95\\% CI] \\\\", "\\midrule"]
    any_row = False
    for d in MESHES:
        n_main = MESHES[d][1]
        rows_b = None
        for n2 in (n_main, 50, 30):
            rows_b = load_rows(d, n2, "ours_v10_noda")
            if rows_b:
                break
        if not rows_b:
            continue
        a, b = matched(d, "ours_v10", "ours_v10_noda", n_main, n2)
        if not a:
            continue
        r_mesh = mesh_radius(d, n_main)
        sa, sb = summarize(a, r_mesh), summarize(b, r_mesh)
        diff = np.array([fl(y, "rot_err_deg") - fl(x, "rot_err_deg") for x, y in zip(a, b)])
        ci = boot_ci(diff)
        lines.append(f"{MESHES[d][0]} & Proposed & {len(a)} & {f1(sa['eR_mean'])} & {pct(sa['sr10'])} & {f1(sa['dd_r_median'],3)} & {pct(sa['wrong_shell'])} & \\\\")
        lines.append(f" & w/o-DA & {len(b)} & {f1(sb['eR_mean'])} & {pct(sb['sr10'])} & {f1(sb['dd_r_median'],3)} & {pct(sb['wrong_shell'])} & {f1(diff.mean())} [{f1(ci[0])}, {f1(ci[1])}] \\\\")
        lines.append("\\midrule")
        any_row = True
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return ("\n".join(lines) + "\n") if any_row else ""


def seed_recall(ds: str, n: int, method: str, ang_thr=20.0, dist_thr=0.25) -> float | None:
    run = RUNS / f"{ds}_{n}gt" / method
    rows = load_rows(ds, n, method)
    if not rows:
        return None
    hits, tot = 0, 0
    for r in rows:
        tag = r["tag"]
        js = run / tag / "seed_00" / f"{tag}_rand_octant_best.json"
        if not js.exists():
            continue
        try:
            d = json.JSONDecoder().raw_decode(js.read_text(encoding="utf-8"))[0]
        except Exception:
            continue
        gt = np.array([fl(r, "gt_dist"), fl(r, "gt_elev_deg"), fl(r, "gt_azim_deg")])
        def unit(e, a):
            e, a = math.radians(e), math.radians(a)
            return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
        ug = unit(gt[1], gt[2])
        ok = False
        for s in d.get("topk_seeds", []):
            ang = math.degrees(math.acos(float(np.clip(ug.dot(unit(s["elev_deg"], s["azim_deg"])), -1, 1))))
            if ang <= ang_thr and abs(s["dist"] - gt[0]) / gt[0] <= dist_thr:
                ok = True
                break
        hits += ok
        tot += 1
    return hits / tot if tot else None


RADIUS_SCHEDULES = (("ours_v10", "Scale-adaptive (proposed)", "Prop"), ("ours_v10_fixedmr", "Fixed multi-radius, 36 shells", "Fixed"), ("ours_v10_1shell", "Single shell at $2r$", "One"))


def radius_blocks(d: str) -> list[tuple[str, str, int, float | None, dict]]:
    """(method, label, key, seed recall, summary) for the three radius schedules on the SAME views (the first views of the N=100 run)."""
    r_mesh = mesh_radius(d, 100)
    n_abl = 30 if (load_rows(d, 30, "ours_v10_fixedmr") or load_rows(d, 30, "ours_v10_1shell")) else 20
    out = []
    for m, label, key in RADIUS_SCHEDULES:
        n_use = 100 if m == "ours_v10" else n_abl
        rows = load_rows(d, n_use, m)
        if m == "ours_v10" and rows:
            rows = sorted(rows, key=lambda r: r["tag"])[:n_abl]
        if not rows:
            continue
        s = summarize(rows, r_mesh)
        s["n"] = len(rows)
        out.append((m, label, key, seed_recall(d, n_use, m), s))
    return out


def radius_table() -> str:
    lines = ["% Auto-generated by tools/export_revision_results.py", "\\begin{table}[!t]", "\\centering",
             "\\caption{Radius-schedule ablation at equal proposal budget ($\\approx 41$k proposals; identical seeds, beam and selector; the same first $N$ views of the $N{=}100$ benchmark for all three schedules). "
             "Seed recall: fraction of views for which at least one of the $K{=}16$ seeds lies within $20^\\circ$ and $\\pm25\\%$ distance of the ground truth.}",
             "\\label{tab:ablation_radius}", "\\normalsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}llccccc@{}}", "\\toprule",
             "Mesh & Schedule & $N$ & seed recall & $e_R$ & SR@10$^\\circ$ & $|\\Delta d|/r$ \\\\", "\\midrule"]
    any_row = False
    for d in ("bunny", "dragon"):
        block = []
        for m, label, key, rec, s in radius_blocks(d):
            block.append(f"{MESHES[d][0] if not block else ''} & {label} & {s['n']} & {pct(rec) if rec is not None else '--'} & {f1(s['eR_mean'])} & {pct(s['sr10'])} & {f1(s['dd_r_median'],3)} \\\\")
        if len(block) >= 2:
            lines += block + ["\\midrule"]
            any_row = True
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return ("\n".join(lines) + "\n") if any_row else ""


def run_machine(ds: str, n: int, method: str) -> str:
    """GPU on which a run was produced: a machine.txt marker is written by finalize_revision when merging worker results."""
    if method == "ours_v10_geom":  # derived offline from the ours_v10 run; same GPU as that run
        method = "ours_v10"
    for nn in (n, 50, 30, 100):
        d = RUNS / f"{ds}_{nn}gt" / method
        if (d / "pose_metrics_summary.csv").exists():
            m = d / "machine.txt"
            return m.read_text(encoding="utf-8").strip() if m.exists() else "RTX 5090"
    return "--"


def runtime_table() -> str:
    lines = ["% Auto-generated by tools/export_revision_results.py", "\\begin{table}[!t]", "\\centering",
             "\\caption{Runtime per view in seconds (wall clock, CUDA-synchronised). On Bunny and Dragon the proposed method and its ablations ran on an RTX~5090 "
             "laptop GPU shared with unrelated jobs and the DR baselines on an RTX~2080\\,Ti (GPU column), so absolute times are comparable only "
             "within a GPU and all times are upper bounds; in the Happy Buddha block the proposed method and the DR baselines ran on the same "
             "RTX~2080\\,Ti (the same-GPU comparison); WAPR, Diff-DOPE and SDFR ran on the RTX~5090 in every block. Proposed, 1/8 budget: $M{=}20$, $S{=}256$, $K{=}8$, $B{=}24$, $L{=}16$ "
             "(both $MS$ and $KBL$ divided by eight) on the first 30 views, on the RTX~2080\\,Ti for Bunny. Total: mean / median / P95; stage columns are means. Global = stage~I proposals; "
             "DA = stage~II sweeps; local = stage~III beams; select = stage~IV; refine = Adam refinement of the DR baselines.}",
             "\\label{tab:runtime}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{@{}lllcccccccc@{}}", "\\toprule",
             "Mesh & Method & GPU & total & median & P95 & global & DA & local & select & refine \\\\", "\\midrule"]
    any_row = False
    for d in ("bunny", "dragon", "happy"):
        n = MESHES[d][1]
        block = []
        for gname, lst in GROUPS:
            for m, _ in (lst + [("ours_v10_fast", "")] if gname == "Proposed" else lst):
                rows = load_rows(d, n, m) or load_rows(d, 50, m) or load_rows(d, 30, m)
                if not rows:
                    continue
                s = summarize(rows, mesh_radius(d, n))
                if "t_mean" not in s:
                    continue
                g = lambda k: f1(s.get(k, float('nan')), 1)
                block.append(f"{MESHES[d][0] if not block else ''} & {SHORT.get(m, m)} & {run_machine(d, n, m)} & {g('t_mean')} & {g('t_median')} & {g('t_p95')} & "
                             f"{g('t_global_s_mean')} & {g('t_dist_align_s_mean')} & {g('t_local_s_mean')} & {g('t_select_s_mean')} & {g('t_refine_s_mean')} \\\\")
        if block:
            lines += block + ["\\midrule"]
            any_row = True
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return ("\n".join(lines) + "\n") if any_row else ""


def pooled_vs(m_a: str, m_b: str) -> dict:
    """SR@10 of two methods on the same views of all five meshes, with an exact McNemar test on the discordant views."""
    b = c = n = ok_a = ok_b = 0
    per_a, per_b = [], []
    for d in MESHES:
        nn = MESHES[d][1]
        ra = {r["tag"]: fl(r, "rot_err_deg") <= 10 for r in load_rows(d, nn, m_a)}
        rb = {r["tag"]: fl(r, "rot_err_deg") <= 10 for r in load_rows(d, nn, m_b)}
        common = sorted(set(ra) & set(rb))
        if not common:
            return {}
        per_a.append(100 * np.mean([ra[t] for t in common]))
        per_b.append(100 * np.mean([rb[t] for t in common]))
        for t in common:
            n += 1
            ok_a += ra[t]
            ok_b += rb[t]
            b += ra[t] and not rb[t]
            c += rb[t] and not ra[t]
    k = b + c
    p = min(1.0, 2 * sum(math.comb(k, i) for i in range(0, min(b, c) + 1)) / 2 ** k) if k else 1.0
    p_txt = "<0.001" if p < 0.001 else f"{p:.3f}"
    return {"n": n, "sr_a": 100 * ok_a / n, "sr_b": 100 * ok_b / n, "b": b, "c": c, "p": p, "p_txt": p_txt,
            "sr_a_min": min(per_a), "sr_a_max": max(per_a), "sr_b_min": min(per_b), "sr_b_max": max(per_b)}


def transfer_stats(m: str) -> dict:
    """Bunny->Dragon change of one method: mean e_R and SR@10 with unpaired bootstrap 95% CIs (views are independent draws)."""
    rb = load_rows("bunny", 100, m) or load_rows("bunny", 50, m) or load_rows("bunny", 30, m)
    rd = load_rows("dragon", 100, m) or load_rows("dragon", 50, m) or load_rows("dragon", 30, m)
    if not rb or not rd:
        return {}
    eb = np.array([fl(r, "rot_err_deg") for r in rb])
    ed = np.array([fl(r, "rot_err_deg") for r in rd])
    rng = np.random.default_rng(zlib.crc32(m.encode()))
    d_mean, d_sr = [], []
    for _ in range(2000):
        a, b = eb[rng.integers(0, len(eb), len(eb))], ed[rng.integers(0, len(ed), len(ed))]
        d_mean.append(b.mean() - a.mean())
        d_sr.append(100 * (np.mean(b <= 10) - np.mean(a <= 10)))
    return {"rb": rb, "rd": rd,
            "dER": float(ed.mean() - eb.mean()), "dER_ci": (float(np.quantile(d_mean, 0.025)), float(np.quantile(d_mean, 0.975))),
            "dSR": float(100 * (np.mean(ed <= 10) - np.mean(eb <= 10))), "dSR_ci": (float(np.quantile(d_sr, 0.025)), float(np.quantile(d_sr, 0.975)))}


def transfer_table() -> str:
    methods = ["ours_v10", "ours_v10_noda", "wapr_full", "diffdope", "sdfr", "grid_adam", "grid_adam_top16", "rand_restart_adam", "adam_dr", "hybrid_dr"]
    lines = ["% Auto-generated by tools/export_revision_results.py", "\\begin{table}[!t]", "\\centering",
             "\\caption{Object transfer with a single configuration: per-mesh mean $e_R$ (deg) and the Bunny$\\to$Dragon change with a bootstrap 95\\% CI "
             "(unpaired, views are independent draws), the same for SR@$10^\\circ$ (percentage points), and $\\Delta|\\Delta d|/r$: change of the median normalised distance error. "
             "All errors are normalised by the mesh radius, so a change reflects the shape of the object, not its size.}",
             "\\label{tab:transfer}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
             "Method & $e_R$ Bunny & $e_R$ Dragon & $\\Delta e_R$ [95\\% CI] & SR@10 B / D & $\\Delta$SR@10 [95\\% CI] & $|\\Delta d|/r$ B / D & $\\Delta|\\Delta d|/r$ \\\\", "\\midrule"]
    any_row = False
    for m in methods:
        ts = transfer_stats(m)
        if not ts:
            continue
        sb, sd = summarize(ts["rb"], mesh_radius("bunny", 100)), summarize(ts["rd"], mesh_radius("dragon", 100))
        lo, hi = ts["dER_ci"]
        slo, shi = ts["dSR_ci"]
        name = "\\textbf{Proposed}" if m == "ours_v10" else SHORT.get(m, m)
        lines.append(f"{name} & {f1(sb['eR_mean'])} & {f1(sd['eR_mean'])} & {f1(ts['dER'])} [{f1(lo)}, {f1(hi)}] & "
                     f"{pct(sb['sr10'])} / {pct(sd['sr10'])} & {ts['dSR']:+.0f} [{slo:+.0f}, {shi:+.0f}] & "
                     f"{f1(sb['dd_r_median'],3)} / {f1(sd['dd_r_median'],3)} & {f1(sd['dd_r_median']-sb['dd_r_median'],3)} \\\\")
        any_row = True
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return ("\n".join(lines) + "\n") if any_row else ""


# ----------------------------------------------------------------------------- figures
def binned_sr_figure(summary: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    methods = [("ours_v10", "Proposed"), ("ours_v10_noda", "w/o-DA"), ("wapr_full", "WAPR (12 hyp.)"), ("grid_adam", "Grid+Adam"), ("rand_restart_adam", "RandRestart+Adam"), ("adam_dr", "Adam-DR")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for col, d in enumerate(("bunny", "dragon")):
        r_mesh = mesh_radius(d, 100)
        for m, label in methods:
            rows = load_rows(d, 100, m) or load_rows(d, 50, m) or load_rows(d, 30, m)
            if not rows:
                continue
            eR = np.array([fl(r, "rot_err_deg") for r in rows])
            elev = np.array([fl(r, "gt_elev_deg") for r in rows])
            dr = np.array([fl(r, "gt_dist") for r in rows]) / r_mesh
            # elevation bins
            eb = [5, 25, 45, 65, 85]
            xs, ys = [], []
            for lo, hi in zip(eb[:-1], eb[1:]):
                sel = (elev >= lo) & (elev < hi)
                if sel.sum() >= 3:
                    xs.append((lo + hi) / 2)
                    ys.append(100 * np.mean(eR[sel] <= 10))
            axes[0, col].plot(xs, ys, marker="o", label=label)
            qs = np.quantile(dr, [0, 0.25, 0.5, 0.75, 1.0])
            xs, ys = [], []
            for lo, hi in zip(qs[:-1], qs[1:]):
                sel = (dr >= lo) & (dr <= hi)
                if sel.sum() >= 3:
                    xs.append((lo + hi) / 2)
                    ys.append(100 * np.mean(eR[sel] <= 10))
            axes[1, col].plot(xs, ys, marker="o", label=label)
        axes[0, col].set_title(MESHES[d][0])
        axes[0, col].set_xlabel("GT elevation (deg)")
        axes[1, col].set_xlabel("GT distance / r (quartile bins)")
        for row in (0, 1):
            axes[row, col].set_ylabel("SR@10° (%)")
            axes[row, col].set_ylim(-3, 103)
            axes[row, col].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_binned_sr.pdf")
    plt.close(fig)


def runtime_figure() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    d = "bunny"
    labels, means, p95s = [], [], []
    for gname, lst in GROUPS:
        for m, _ in lst:
            rows = load_rows(d, 100, m) or load_rows(d, 50, m) or load_rows(d, 30, m)
            if not rows:
                continue
            s = summarize(rows, mesh_radius(d, 100))
            if "t_mean" in s:
                labels.append(SHORT.get(m, m))
                means.append(s["t_mean"])
                p95s.append(s["t_p95"])
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    x = np.arange(len(labels))
    ax.bar(x, means, color="#4c72b0", label="mean")
    ax.plot(x, p95s, "k_", markersize=14, label="P95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("seconds / view")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_runtime.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- correlations
def correlation_analysis() -> dict:
    """Spearman correlations between per-view e_R of the proposed method and view factors."""
    try:
        from scipy.stats import spearmanr
    except Exception:
        return {}
    out = {}
    for d in ("bunny", "dragon"):
        rows = load_rows(d, 100, "ours_v10")
        if len(rows) < 10:
            continue
        r_mesh = mesh_radius(d, 100)
        eR = np.array([fl(r, "rot_err_deg") for r in rows])
        elev = np.array([fl(r, "gt_elev_deg") for r in rows])
        dr = np.array([fl(r, "gt_dist") for r in rows]) / r_mesh
        fill = []
        for r in rows:
            npz = RUNS / f"{d}_100gt" / "ours_v10" / r["tag"] / "gt_refs_search.npz"
            try:
                a = np.load(npz)["alpha"]
                fill.append(float((a > 0.5).mean()))
            except Exception:
                fill.append(float("nan"))
        fill = np.array(fill)
        res = {"n": len(rows)}
        for name, x in (("elev", elev), ("dist_ratio", dr), ("fill", fill)):
            ok = ~np.isnan(x)
            if ok.sum() > 5:
                rho, p = spearmanr(x[ok], eR[ok])
                res[name] = {"rho": float(rho), "p": float(p)}
        out[d] = res
    return out


# ----------------------------------------------------------------------------- correlation table
def corr_table(corr: dict) -> str:
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{Spearman rank correlation between the per-view rotation error of the proposed method and three view factors "
             "(ground-truth elevation, distance ratio $d/r$ and silhouette fill ratio). $p$-values are two-sided; $N$ views per mesh.}",
             "\\label{tab:corr}", "\\begin{tabular}{lcccc}", "\\toprule",
             "Mesh & $N$ & elevation $\\rho$ ($p$) & $d/r$ $\\rho$ ($p$) & fill $\\rho$ ($p$) \\\\", "\\midrule"]
    any_row = False
    for d in ("bunny", "dragon"):
        res = corr.get(d)
        if not res:
            continue
        any_row = True
        cells = []
        for name in ("elev", "dist_ratio", "fill"):
            v = res.get(name)
            cells.append(f"{v['rho']:+.2f} ({v['p']:.3f})" if v else "--")
        lines.append(f"{MESHES[d][0]} & {res['n']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return ("\n".join(lines) + "\n") if any_row else ""


STUB_CAPTIONS = {
    "tab_results_main.tex": ("tab:bunny100gt_main", "Bunny / Dragon, $N=100$ views: results pending (run in progress)."),
    "tab_results_bunny100gt.tex": ("tab:bunny100gt_main", "Bunny, $N=100$ views: results pending (run in progress)."),
    "tab_results_dragon100gt.tex": ("tab:dragon100gt_main", "Dragon, $N=100$ views: results pending (run in progress)."),
    "tab_new_meshes.tex": ("tab:new_meshes", "Armadillo / Happy Buddha / Lucy: results pending."),
    "tab_cross_object.tex": ("tab:cross_object", "Cross-object summary: results pending."),
    "tab_ablation_da.tex": ("tab:ablation_da", "Distance-alignment ablation: results pending."),
    "tab_ablation_radius.tex": ("tab:ablation_radius", "Radius-schedule ablation: results pending."),
    "tab_runtime.tex": ("tab:runtime", "Runtime: results pending."),
    "tab_transfer.tex": ("tab:transfer", "Transfer: results pending."),
    "tab_corr.tex": ("tab:corr", "Correlation analysis: results pending."),
}


def fit_width(tex: str) -> str:
    """Scale a tabular down to the text width when it is too wide (adjustbox; never scales up)."""
    return tex.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}", 1)\
              .replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}", 1)


def stub_table(name: str) -> str:
    label, cap = STUB_CAPTIONS[name]
    return ("\\begin{table}[t]\\centering\\small\\caption{%s}\\label{%s}\\begin{tabular}{l}\\toprule PENDING \\\\ \\bottomrule\\end{tabular}\\end{table}\n" % (cap, label))


# ----------------------------------------------------------------------------- macros
def macros(summary: dict, corr: dict) -> str:
    def m(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}"
    L = ["% Auto-generated by tools/export_revision_results.py -- numbers used in the text"]
    for d in MESHES:
        key = d.capitalize().replace("_", "")
        n = MESHES[d][1]
        s = summary.get(d, {}).get("ours_v10", {})
        if s:
            L += [m(f"Res{key}ER", f1(s["eR_mean"])), m(f"Res{key}ERmed", f1(s["eR_median"])), m(f"Res{key}SRten", f"{100*s['sr10']:.0f}"),
                  m(f"Res{key}SRfive", f"{100*s['sr5']:.0f}"), m(f"Res{key}N", str(s["n"])), m(f"Res{key}DDr", f1(s["dd_r_median"], 3)),
                  m(f"Res{key}CIlo", f1(s["eR_ci"][0])), m(f"Res{key}CIhi", f1(s["eR_ci"][1]))]
            if "t_median" in s:
                L.append(m(f"Res{key}Time", f1(s["t_median"], 0)))
        for alt, suffix in (("grid_adam", "Grid"), ("rand_restart_adam", "RR"), ("adam_dr", "Adam"), ("ours_v10_noda", "NoDA"), ("ours_v10_geom", "Geom"), ("greedy_bt_local", "Greedy"), ("rand_topk_only", "TopK"),
                            ("wapr_full", "WaprFull"), ("wapr_refine", "WaprRef"), ("diffdope", "DiffDope"), ("sdfr", "Sdfr"), ("hybrid_dr", "Hybrid"),
                            ("ours_v10_fast", "Fast"), ("foundationpose_full", "FpFull"), ("foundationpose_refine", "FpRef"),
                            ("ours_v10_stage1", "StageOne")):
            s2 = summary.get(d, {}).get(alt, {})
            if s2:
                L += [m(f"Res{key}{suffix}ER", f1(s2["eR_mean"])), m(f"Res{key}{suffix}SRten", f"{100*s2['sr10']:.0f}"), m(f"Res{key}{suffix}N", str(s2["n"])),
                      m(f"Res{key}{suffix}DDr", f1(s2["dd_r_median"], 3)), m(f"Res{key}{suffix}Wrong", f"{100*s2['wrong_shell']:.0f}"),
                      m(f"Res{key}{suffix}ERmed", f1(s2["eR_median"]))]
                if "t_median" in s2:
                    L.append(m(f"Res{key}{suffix}Time", f1(s2["t_median"], 1 if s2["t_median"] < 10 else 0)))
    st_n = st_ok = 0
    for d in MESHES:
        rws = load_rows(d, MESHES[d][1], "ours_v10_stage1")
        st_n += len(rws)
        st_ok += sum(fl(row, "rot_err_deg") <= 10 for row in rws)
    if st_n:
        L += [m("ResPoolStageOneN", str(st_n)), m("ResPoolStageOneOk", str(st_ok)), m("ResPoolStageOneSRten", f"{100 * st_ok / st_n:.1f}")]
    for d, res in corr.items():
        key = d.capitalize()
        for name, v in res.items():
            if isinstance(v, dict):
                L += [m(f"Corr{key}{name.replace('_','')}Rho", f"{v['rho']:.2f}"), m(f"Corr{key}{name.replace('_','')}P", f"{v['p']:.3f}")]
    for alt, suffix in (("wapr_full", "WaprFull"), ("wapr_refine", "WaprRef"), ("diffdope", "DiffDope"), ("sdfr", "Sdfr"),
                        ("foundationpose_full", "FpFull"), ("foundationpose_refine", "FpRef")):
        ps = pooled_vs("ours_v10", alt)
        if ps:
            L += [m(f"ResPool{suffix}N", str(ps["n"])), m(f"ResPool{suffix}OursSRten", f"{ps['sr_a']:.1f}"), m(f"ResPool{suffix}SRten", f"{ps['sr_b']:.1f}"),
                  m(f"ResPool{suffix}OnlyOurs", str(ps["b"])), m(f"ResPool{suffix}OnlyOther", str(ps["c"])), m(f"ResPool{suffix}P", ps["p_txt"]),
                  m(f"ResPool{suffix}SRlo", f"{ps['sr_b_min']:.0f}"), m(f"ResPool{suffix}SRhi", f"{ps['sr_b_max']:.0f}"),
                  m(f"ResPool{suffix}OursSRlo", f"{ps['sr_a_min']:.0f}"), m(f"ResPool{suffix}OursSRhi", f"{ps['sr_a_max']:.0f}")]
    for alt, suffix in (("wapr_sanity", "WaprSanity"), ("diffdope_sanity", "DiffDopeSanity"), ("sdfr_sanity", "SdfrSanity"),
                        ("foundationpose_sanity", "FpSanity")):
        ok, close = [], []
        for d in MESHES:
            nn = MESHES[d][1]
            r_d = mesh_radius(d, nn)
            for row in load_rows(d, nn, alt):
                ok.append(fl(row, "rot_err_deg") <= 10)
                close.append(fl(row, "gt_dist") / r_d < 2.0)
        if ok:
            ok, close = np.array(ok), np.array(close)
            L += [m(f"Res{suffix}N", str(len(ok))), m(f"Res{suffix}SRten", f"{100 * ok.mean():.0f}"),
                  m(f"Res{suffix}CloseN", str(int(close.sum()))),
                  m(f"Res{suffix}CloseSRten", f"{100 * ok[close].mean():.0f}" if close.any() else "--"),
                  m(f"Res{suffix}FarSRten", f"{100 * ok[~close].mean():.0f}" if (~close).any() else "--")]
    fill_med, all_dr, all_fill, fail_dr, fail_fill = {}, [], [], [], []
    for d in MESHES:
        nn = MESHES[d][1]
        r_d = mesh_radius(d, nn)
        ours = {row["tag"]: row for row in load_rows(d, nn, "ours_v10")}
        fills = {}
        for t in ours:
            f = RUNS / f"{d}_{nn}gt" / "ours_v10" / t / "gt_refs_search.npz"
            if f.exists():
                fills[t] = float((np.load(f)["alpha"] > 0.5).mean())
        if fills:
            fill_med[d] = float(np.median(list(fills.values())))
        for row in load_rows(d, nn, "wapr_full"):
            t = row["tag"]
            if t not in ours or t not in fills:
                continue
            drv = fl(ours[t], "gt_dist") / r_d
            all_dr.append(drv)
            all_fill.append(fills[t])
            if fl(row, "rot_err_deg") > 10:
                fail_dr.append(drv)
                fail_fill.append(fills[t])
    for d, v in fill_med.items():
        L.append(m(f"Res{d.capitalize()}FillMed", f"{100 * v:.0f}"))
    if fail_dr:
        L += [m("ResWaprFailN", str(len(fail_dr))), m("ResWaprFailDrMed", f"{np.median(fail_dr):.2f}"),
              m("ResWaprFailFillMed", f"{100 * np.median(fail_fill):.0f}"), m("ResWaprFailInside", str(int(np.sum(np.array(fail_dr) < 1.0)))),
              m("ResAllDrMed", f"{np.median(all_dr):.2f}"), m("ResAllFillMed", f"{100 * np.median(all_fill):.0f}")]
    for d in ("bunny", "dragon"):
        fast = {row["tag"]: row for row in load_rows(d, 100, "ours_v10_fast")}
        if fast:
            ref = [row for row in load_rows(d, 100, "ours_v10") if row["tag"] in fast]
            sref = summarize(ref, mesh_radius(d, 100))
            key = d.capitalize()
            L += [m(f"Res{key}FastRefSRten", f"{100 * sref['sr10']:.0f}"), m(f"Res{key}FastRefER", f1(sref["eR_mean"])),
                  m(f"Res{key}FastRefERmed", f1(sref["eR_median"])), m(f"Res{key}FastRefTime", f1(sref.get("t_median", float("nan")), 0))]
            rr = [row for row in load_rows(d, 100, "rand_restart_adam") if row["tag"] in fast]
            if len(rr) == len(fast):
                srr = summarize(rr, mesh_radius(d, 100))
                L += [m(f"Res{key}FastRRSRten", f"{100 * srr['sr10']:.0f}"), m(f"Res{key}FastRRTime", f1(srr.get("t_median", float("nan")), 0))]
    ts = transfer_stats("rand_restart_adam")
    if ts:
        L += [m("ResTransferRRdER", f1(ts["dER"])), m("ResTransferRRdERlo", f1(ts["dER_ci"][0])), m("ResTransferRRdERhi", f1(ts["dER_ci"][1])),
              m("ResTransferRRdSR", f"{ts['dSR']:.0f}"), m("ResTransferRRdSRlo", f"{ts['dSR_ci'][0]:.0f}"), m("ResTransferRRdSRhi", f"{ts['dSR_ci'][1]:.0f}")]
    # radius-schedule ablation (matched views)
    for d in ("bunny", "dragon"):
        for m_, label, key, rec, s in radius_blocks(d):
            k = f"Rad{d.capitalize()}{key}"
            L += [m(f"{k}ER", f1(s["eR_mean"])), m(f"{k}SRten", f"{100*s['sr10']:.0f}"), m(f"{k}DDr", f1(s["dd_r_median"], 3)),
                  m(f"{k}N", str(s["n"])), m(f"{k}Wrong", f"{100*s['wrong_shell']:.0f}")]
            if rec is not None:
                L.append(m(f"{k}Rec", f"{100*rec:.0f}"))
    # Placeholders for every macro the text may use, so the manuscript compiles before all runs finish.
    defined = {ln.split("{")[1].split("}")[0] for ln in L if ln.startswith("\\newcommand")}
    expected = []
    for d in ("Bunny", "Dragon"):
        for key in ("Prop", "Fixed", "One"):
            expected += [f"Rad{d}{key}{s}" for s in ("ER", "SRten", "DDr", "N", "Wrong", "Rec")]
    for d in MESHES:
        key = d.capitalize().replace("_", "")
        expected += [f"Res{key}{s}" for s in ("ER", "ERmed", "SRten", "SRfive", "N", "DDr", "CIlo", "CIhi", "Time")]
        for suffix in ("Grid", "RR", "Adam", "NoDA", "Geom", "Greedy", "TopK", "WaprFull", "WaprRef", "DiffDope", "Sdfr", "Hybrid"):
            expected += [f"Res{key}{suffix}{s}" for s in ("ER", "SRten", "N", "DDr", "Wrong", "Time")]
    for d in ("Bunny", "Dragon"):
        for name in ("elev", "distratio", "fill"):
            expected += [f"Corr{d}{name}Rho", f"Corr{d}{name}P"]
    for suffix in ("WaprFull", "WaprRef", "DiffDope", "Sdfr"):
        expected += [f"ResPool{suffix}{x}" for x in ("N", "OursSRten", "SRten", "OnlyOurs", "OnlyOther", "P", "SRlo", "SRhi", "OursSRlo", "OursSRhi")]
    for suffix in ("WaprSanity", "DiffDopeSanity", "SdfrSanity"):
        expected += [f"Res{suffix}{x}" for x in ("N", "SRten", "CloseN", "CloseSRten", "FarSRten")]
    expected += [f"Res{d.capitalize()}FillMed" for d in MESHES]
    expected += ["ResWaprFailN", "ResWaprFailDrMed", "ResWaprFailFillMed", "ResWaprFailInside", "ResAllDrMed", "ResAllFillMed"]
    for d in ("Bunny", "Dragon"):
        expected += [f"Res{d}Fast{x}" for x in ("ER", "SRten", "N", "DDr", "Wrong", "ERmed", "Time")]
        expected += [f"Res{d}FastRef{x}" for x in ("SRten", "ER", "ERmed", "Time")]
        expected += [f"Res{d}FastRR{x}" for x in ("SRten", "Time")]
    L.append("% placeholders (no data yet)")
    for name in expected:
        if ("\\" + name) not in defined:
            L.append(f"\\providecommand{{\\{name}}}{{\\textbf{{??}}}}")
    return "\n".join(L) + "\n"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for d in MESHES:
        n = MESHES[d][1]
        summary[d] = {}
        for gname, lst in GROUPS + [("", [("ours_v10_fixedmr", ""), ("ours_v10_1shell", ""), ("ours_v10_fast", "")])]:
            for mth, _ in lst:
                rows = load_rows(d, n, mth) or load_rows(d, 50, mth) or load_rows(d, 30, mth)
                if rows:
                    summary[d][mth] = summarize(rows, mesh_radius(d, n))
    corr = correlation_analysis()
    for name, fn in (("tab_results_bunny100gt.tex", lambda: main_table("bunny")), ("tab_results_dragon100gt.tex", lambda: main_table("dragon")),
                     ("tab_results_main.tex", main_table_compact), ("tab_results_full.tex", main_table_combined),
                     ("tab_new_meshes.tex", new_meshes_table), ("tab_cross_object.tex", cross_object_table), ("tab_ablation_da.tex", da_table),
                     ("tab_ablation_radius.tex", radius_table), ("tab_runtime.tex", runtime_table), ("tab_transfer.tex", transfer_table),
                     ("tab_corr.tex", lambda: corr_table(corr))):
        txt = fn()
        if txt and name in SUPP_TABLES:
            txt = txt.replace("\\begin{table}[!t]", "\\begin{table}[!htbp]").replace("\\begin{table}[t]", "\\begin{table}[!htbp]")
        if txt and "PENDING" not in txt and txt.count("\\\\") >= 2:
            (TABLES / name).write_text(fit_width(txt), encoding="utf-8")
            print(f"[table] {name}")
        else:
            (TABLES / name).write_text(stub_table(name), encoding="utf-8")
            print(f"[table] {name}: no data yet (stub written)")
    (MAN / "results_macros.tex").write_text(macros(summary, corr), encoding="utf-8")
    binned_sr_figure(summary)
    runtime_figure()
    (ROOT / "benchmark_results").mkdir(exist_ok=True)
    (ROOT / "benchmark_results" / "revision_summary.json").write_text(json.dumps({"summary": summary, "correlations": corr}, indent=2, default=str), encoding="utf-8")
    for d in summary:
        for mth, s in summary[d].items():
            print(f"{d:10s} {mth:20s} n={s['n']:3d} eR={s['eR_mean']:7.2f} SR10={100*s['sr10']:5.1f}% ddr={s['dd_r_median']:.3f}" + (f" t={s['t_median']:.0f}s" if 't_median' in s else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
