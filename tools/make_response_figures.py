#!/usr/bin/env python3
"""
Figures for the response letter.

offset   SR@10 deg (symmetry-aware) of the refiners against the initial rotation offset on the BOP frames, the
         initialisation-free methods as horizontal references, and the refiners started from the proposed estimate.
         Same records and statistics as Table 4 / Supplementary Tables S7 and S11.
novelty  The three controlled tests of the novelty argument, one panel per cause of failure (basin commitment, radial
         flatness, premature appearance). Every number is read from the generated macro files, from the generated
         distance-alignment table and from the supplementary text, i.e. from what the manuscript itself prints
         (without the manuscript sources: from lm_dist_alignment_results_metric/dist_alignment.csv for the shell
         selection; the macro files and the table are written by tools/export_revision_results.py and
         tools/analyze_fromours.py).

  python tools/make_response_figures.py                  # both figures
  python tools/make_response_figures.py --fig novelty    # only one
Writes manuscript/computational_visual_media/figures/response/fig_r_{offset,novelty}.pdf
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp_bop", Path(__file__).resolve().parent / "export_bop_results.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)
RES = ROOT / "bop6dof_results" if (ROOT / "bop6dof_results").exists() else ROOT / "results" / "bop6dof_results"
eb.RES = RES
OUT = ROOT / "manuscript" / "computational_visual_media" / "figures" / "response"

REFINERS = [("baseline_wapr_refine.jsonl", "WAPR refine", "#1f77b4", "o"),
            ("baseline_foundationpose_refine.jsonl", "FoundationPose refine", "#8c564b", "P"),
            ("baseline_megapose_refine.jsonl", "MegaPose refine", "#e377c2", "X"),
            ("baseline_gotrack_refine.jsonl", "GoTrack", "#17becf", "h"),
            ("baseline_sdfr.jsonl", "SDFR", "#2ca02c", "s"),
            ("baseline_diffdope.jsonl", "Diff-DOPE", "#ff7f0e", "^"),
            ("baseline_scflow2.jsonl", "SCFlow2", "#9467bd", "v"),
            ("results_init_bins.jsonl", "Proposed, local refinement", "#d62728", "D")]
FROM_OURS = {"WAPR refine": "baseline_wapr_from_ours.jsonl", "FoundationPose refine": "baseline_foundationpose_from_ours.jsonl",
             "MegaPose refine": "baseline_megapose_from_ours.jsonl", "GoTrack": "baseline_gotrack_from_ours.jsonl",
             "SDFR": "baseline_sdfr_from_ours.jsonl",
             "Diff-DOPE": "baseline_diffdope_from_ours.jsonl"}


def load(ds: str, fn: str) -> list[dict]:
    uniq = {}
    for r in eb.read_jsonl(RES / ds / fn):
        if "e_R_deg" in r:
            uniq[(r.get("tag"), str(r.get("bin", "none")))] = r
    out = list(uniq.values())
    eb.add_sym_error(ds, out)
    return out


def sr(recs: list[dict]) -> float:
    s = eb.summarize(recs)
    return 100.0 * s["sr10_sym"] if s else float("nan")


def set_style() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "mathtext.fontset": "stix"})


def fig_offset() -> int:
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), sharey=True)
    xb = [0, 1, 2, 3]
    xo = 4.35
    for ax, (ds, name) in zip(axes, eb.DATASETS):
        ours = sr(load(ds, "results.jsonl"))
        wapr_full = sr(load(ds, "baseline_wapr_full.jsonl"))
        ax.axhline(wapr_full, color="0.45", ls=":", lw=1.0, zorder=1)
        ax.axhline(sr(load(ds, "baseline_foundationpose_full.jsonl")), color="#8c564b", ls="-.", lw=0.9, zorder=1)
        mp_full = load(ds, "baseline_megapose_full.jsonl")
        if mp_full:
            ax.axhline(sr(mp_full), color="#e377c2", ls=(0, (3, 1, 1, 1)), lw=0.9, zorder=1)
        ax.axhline(ours, color="k", ls="--", lw=1.0, zorder=1)
        for fn, label, col, mk in REFINERS:
            recs = load(ds, fn)
            if not recs:
                continue
            ys = [sr([r for r in recs if str(r.get("bin")) == b]) for b in eb.BINS]
            ls = "-" if not label.startswith("Proposed") else (0, (4, 1.5))
            ax.plot(xb, ys, color=col, marker=mk, ms=3.2, lw=1.1, ls=ls, label=label, zorder=3)
            if label in FROM_OURS:
                y1 = sr(load(ds, FROM_OURS[label]))
                ax.plot([3, xo], [ys[-1], y1], color=col, lw=0.8, ls=(0, (1, 1.2)), zorder=2)
                ax.plot([xo], [y1], color=col, marker="*", ms=7.5, mec="k", mew=0.4, zorder=4)
        ax.set_title(name, fontsize=8.5, pad=3)
        ax.set_xticks(xb + [xo])
        ax.set_xticklabels(["15-30", "30-60", "60-90", "90-180", "from\nproposed"], fontsize=7)
        ax.set_xlim(-0.35, xo + 0.45)
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", lw=0.3, color="0.85")
        ax.tick_params(length=2.5, pad=1.5)
    axes[0].set_ylabel("SR@10\u00b0 sym. (%)")
    fig.text(0.47, 0.035, "initial rotation offset of $T_0$ (deg)", ha="center", fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="k", ls="--", lw=1.0), plt.Line2D([], [], color="0.45", ls=":", lw=1.0),
                plt.Line2D([], [], color="#8c564b", ls="-.", lw=0.9), plt.Line2D([], [], color="#e377c2", ls=(0, (3, 1, 1, 1)), lw=0.9),
                plt.Line2D([], [], color="0.3", marker="*", ms=7.5, mec="k", mew=0.4, ls="none")]
    labels += ["Proposed, no $T_0$", "WAPR 12 hyp., no $T_0$", "FoundationPose, no $T_0$", "MegaPose, no $T_0$",
               "refiner started from the proposed estimate"]
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=6.6, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=2.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.2, top=0.75, wspace=0.08)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_r_offset.pdf")
    fig.savefig(OUT / "fig_r_offset.png", dpi=200)
    print("[written]", OUT / "fig_r_offset.pdf")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# novelty figure: numbers come from the files the manuscript is typeset from
# ---------------------------------------------------------------------------------------------------------------------
MS = ROOT / "manuscript" / "computational_visual_media"
MESHES = ["Bunny", "Dragon", "Armadillo", "Happy", "Lucy"]
C_FULL, C_CTRL, C_CTRL2 = "#b2182b", "0.55", "0.80"


def read_macros() -> dict[str, str]:
    """All \\newcommand / \\providecommand values of the generated *_macros.tex files."""
    pat = re.compile(r"\\(?:newcommand|providecommand)\{\\([A-Za-z]+)\}\{([^{}]*)\}")
    out: dict[str, str] = {}
    for fn in sorted(MS.glob("*_macros.tex")):
        for name, val in pat.findall(fn.read_text(encoding="utf-8")):
            out.setdefault(name, val)
    return out


def read_da_table() -> dict[str, dict[str, float | str]]:
    """Mean e_R with and without the distance alignment and the paired CI, from the generated manuscript table."""
    txt = (MS / "tables" / "tab_ablation_da.tex").read_text(encoding="utf-8")
    res: dict[str, dict[str, float | str]] = {}
    for mesh in ("Bunny", "Dragon"):
        m = re.search(mesh + r" & Proposed & (\d+) & ([\d.]+) .*?\\\\\s*& w/o-DA & (\d+) & ([\d.]+) .*? & ([\d.]+) (\[[\d., ]+\]) \\\\",
                      txt, flags=re.S)
        if not m:
            raise RuntimeError(f"tab_ablation_da.tex: rows of {mesh} not found")
        res[mesh] = {"n": int(m.group(1)), "full": float(m.group(2)), "noda": float(m.group(4)), "delta": float(m.group(5)),
                     "ci": m.group(6)}
    return res


def read_shell_selection() -> dict[str, float]:
    """Correct-shell rates on Linemod sensor depth.

    Read from the supplementary paragraph that reports them; where the manuscript sources are not present (code
    release), from the per-frame output of run_lm_dist_alignment_eval.py. If both exist they must agree.
    """
    from_text = from_csv = None
    supp = MS / "supplementary_pr.tex"
    if supp.exists():
        txt = " ".join(supp.read_text(encoding="utf-8").split())
        m = re.search(r"of (\d+) LM frames.*?correct shell \(\$\\pm15\\%\$\) in (\d+)\\% of the frames \((?:mean|median) distance "
                      r"error (\d+)\\%\), the depth-guided selection in (\d+)\\%", txt)
        if not m:
            raise RuntimeError("supplementary_pr.tex: shell-selection sentence not found")
        from_text = {"n": int(m.group(1)), "without": float(m.group(2)), "with": float(m.group(4))}
    for base in (ROOT, ROOT / "results"):
        p = base / "lm_dist_alignment_results_metric" / "dist_alignment.csv"
        if p.exists():
            with p.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            share = lambda k: float(round(100.0 * sum(r[k].strip().lower() == "true" for r in rows) / len(rows)))  # noqa: E731
            from_csv = {"n": len(rows), "without": share("correct_shell_geom"), "with": share("correct_shell_guided")}
            break
    if from_text and from_csv and from_text != from_csv:
        raise RuntimeError(f"shell selection: supplementary text {from_text} and dist_alignment.csv {from_csv} disagree")
    if not (from_text or from_csv):
        raise FileNotFoundError("shell-selection numbers: neither supplementary_pr.tex nor "
                                "lm_dist_alignment_results_metric/dist_alignment.csv (python run_lm_dist_alignment_eval.py)")
    return from_text or from_csv


def label_bars(ax, bars, fmt="{:.0f}", dy=1.5, fs=5.6):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=fs, color="0.15")


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", lw=0.3, color="0.88")
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=1.5)


def fig_novelty() -> int:
    set_style()
    needed = [MS / "results_macros.tex", MS / "fromours_macros.tex", MS / "tables" / "tab_ablation_da.tex"]
    missing = [str(p.relative_to(ROOT)) for p in needed if not p.exists()]
    if missing:
        print("[skipped] novelty figure: generated files not found (" + ", ".join(missing) + "); run "
              "tools/export_revision_results.py and tools/analyze_fromours.py first")
        return 1
    try:
        shell = read_shell_selection()
    except FileNotFoundError as e:
        print("[skipped] novelty figure:", e)
        return 1
    mac = read_macros()
    da = read_da_table()
    val = lambda k: float(mac[k])  # noqa: E731

    fig = plt.figure(figsize=(6.6, 2.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[5.9, 2.0, 1.15, 3.4], left=0.06, right=0.995, bottom=0.2, top=0.74, wspace=0.44)
    ax_a, ax_b1, ax_b2, ax_c = (fig.add_subplot(gs[0, i]) for i in range(4))

    # (a) basin commitment: same proposals, three ways of using them
    x = list(range(len(MESHES)))
    w = 0.27
    series = [("TopKSRten", "best proposal, no refinement (Rand-TopK)", C_CTRL2), ("GreedySRten", "width-one walk (Greedy-BT)", C_CTRL),
              ("SRten", "beam (proposed)", C_FULL)]
    for i, (suffix, lab, col) in enumerate(series):
        ys = [val(f"Res{m}{suffix}") for m in MESHES]
        bars = ax_a.bar([xi + (i - 1) * w for xi in x], ys, w * 0.94, color=col, label=lab, zorder=3)
        label_bars(ax_a, bars)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Bunny", "Dragon", "Armadillo", "Happy B.", "Lucy"], fontsize=6.4)
    ax_a.set_ylabel("SR@10\u00b0 (%)")
    ax_a.set_ylim(0, 112)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    tidy(ax_a)

    # (b) radial flatness: distance alignment removed (matched synthetic views) and isolated on sensor depth
    xb = [0, 1]
    wb = 0.36
    full = [da["Bunny"]["full"], da["Dragon"]["full"]]
    noda = [da["Bunny"]["noda"], da["Dragon"]["noda"]]
    label_bars(ax_b1, ax_b1.bar([v - wb / 2 for v in xb], noda, wb * 0.94, color=C_CTRL, zorder=3), "{:.2f}", 0.03)
    label_bars(ax_b1, ax_b1.bar([v + wb / 2 for v in xb], full, wb * 0.94, color=C_FULL, zorder=3), "{:.2f}", 0.03)
    ax_b1.set_xticks(xb)
    ax_b1.set_xticklabels([f"Bunny\n(N={da['Bunny']['n']})", f"Dragon\n(N={da['Dragon']['n']})"], fontsize=6.6)
    ax_b1.set_ylabel("mean $e_R$ (deg)", labelpad=1.5)
    ax_b1.set_ylim(0, max(noda) * 1.22)
    tidy(ax_b1)
    label_bars(ax_b2, ax_b2.bar([0], [shell["without"]], 0.7, color=C_CTRL, zorder=3))
    label_bars(ax_b2, ax_b2.bar([1], [shell["with"]], 0.7, color=C_FULL, zorder=3))
    ax_b2.set_xticks([0.5])
    ax_b2.set_xticklabels([f"Linemod\n(N={shell['n']})"], fontsize=6.6)
    ax_b2.set_xlim(-0.6, 1.6)
    ax_b2.set_ylabel("correct shell (%)", labelpad=1.0)
    ax_b2.set_ylim(0, 112)
    ax_b2.set_yticks([0, 50, 100])
    tidy(ax_b2)

    # (c) premature appearance: Diff-DOPE without and with its colour term, per initial-offset bin (LM-O + LM)
    xc = [0, 1, 2, 3]
    wc = 0.38
    off = [val(f"BopRgbPoolBin{b}Off") for b in "ABCD"]
    on = [val(f"BopRgbPoolBin{b}On") for b in "ABCD"]
    label_bars(ax_c, ax_c.bar([v - wc / 2 for v in xc], off, wc * 0.94, color=C_FULL, zorder=3))
    label_bars(ax_c, ax_c.bar([v + wc / 2 for v in xc], on, wc * 0.94, color=C_CTRL, zorder=3))
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(["15-30", "30-60", "60-90", "90-180"], fontsize=6.8)
    ax_c.set_xlabel("initial rotation offset (deg)", fontsize=7, labelpad=1.5)
    ax_c.set_ylabel("SR@10\u00b0 sym. (%)", labelpad=1.5)
    ax_c.set_ylim(0, 112)
    ax_c.set_yticks([0, 25, 50, 75, 100])
    tidy(ax_c)

    # panel titles and legends above the axes
    def centre(*axes):
        boxes = [a.get_position() for a in axes]
        return 0.5 * (min(b.x0 for b in boxes) + max(b.x1 for b in boxes))

    for axes, title in (((ax_a,), "(a) Basin commitment"), ((ax_b1, ax_b2), "(b) Radial flatness"), ((ax_c,), "(c) Premature appearance")):
        fig.text(centre(*axes), 0.965, title, ha="center", va="top", fontsize=8.5, fontweight="bold")
    patch = lambda col: plt.Rectangle((0, 0), 1, 1, color=col)  # noqa: E731
    kw = dict(fontsize=6.1, frameon=False, handlelength=1.1, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.25)
    fig.legend([patch(C_CTRL2), patch(C_CTRL), patch(C_FULL)], ["best proposal, no refinement", "width-one walk", "beam (proposed)"],
               loc="upper center", bbox_to_anchor=(centre(ax_a), 0.895), ncol=3, columnspacing=0.9, **kw)
    fig.legend([patch(C_CTRL), patch(C_FULL)], ["without distance alignment", "with it (proposed)"],
               loc="upper center", bbox_to_anchor=(centre(ax_b1, ax_b2), 0.895), ncol=1, **kw)
    fig.legend([patch(C_FULL), patch(C_CTRL)], ["mask + depth, no colour", "colour from the first iteration"],
               loc="upper center", bbox_to_anchor=(centre(ax_c), 0.895), ncol=1, **kw)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_r_novelty.pdf")
    fig.savefig(OUT / "fig_r_novelty.png", dpi=200)
    print("[written]", OUT / "fig_r_novelty.pdf")
    print("[data] basin   :", {m: [mac[f"Res{m}{s}"] for s in ("TopKSRten", "GreedySRten", "SRten")] for m in MESHES})
    print("[data] radial  :", da, shell)
    print("[data] colour  :", list(zip(off, on)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", choices=["offset", "novelty", "all"], default="all")
    args = ap.parse_args()
    rc = 0
    if args.fig in ("offset", "all"):
        rc |= fig_offset()
    if args.fig in ("novelty", "all"):
        rc |= fig_novelty()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
