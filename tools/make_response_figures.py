#!/usr/bin/env python3
"""
Figure for the response letter: SR@10 deg (symmetry-aware) of the refiners against the initial rotation offset on the
BOP frames, the initialisation-free methods as horizontal references, and the refiners started from the proposed
estimate. Same records and statistics as Table 4 / Supplementary Tables S7 and S11.

  python tools/make_response_figures.py
Writes manuscript/computational_visual_media/figures/response/fig_r_offset.pdf
"""
from __future__ import annotations

import importlib.util
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
            ("baseline_sdfr.jsonl", "SDFR", "#2ca02c", "s"),
            ("baseline_diffdope.jsonl", "Diff-DOPE", "#ff7f0e", "^"),
            ("baseline_scflow2.jsonl", "SCFlow2", "#9467bd", "v"),
            ("results_init_bins.jsonl", "Proposed II-IV (local stages)", "#d62728", "D")]
FROM_OURS = {"WAPR refine": "baseline_wapr_from_ours.jsonl", "SDFR": "baseline_sdfr_from_ours.jsonl",
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


def main() -> int:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), sharey=True)
    xb = [0, 1, 2, 3]
    xo = 4.35
    for ax, (ds, name) in zip(axes, eb.DATASETS):
        ours = sr(load(ds, "results.jsonl"))
        wapr_full = sr(load(ds, "baseline_wapr_full.jsonl"))
        ax.axhline(wapr_full, color="0.45", ls=":", lw=1.0, zorder=1)
        ax.axhline(ours, color="k", ls="--", lw=1.0, zorder=1)
        for fn, label, col, mk in REFINERS:
            recs = load(ds, fn)
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
                plt.Line2D([], [], color="0.3", marker="*", ms=7.5, mec="k", mew=0.4, ls="none")]
    labels += ["Proposed, no $T_0$", "WAPR 12 hyp., no $T_0$", "refiner started from the proposed estimate"]
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, 1.02),
               handlelength=2.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.2, top=0.79, wspace=0.08)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_r_offset.pdf")
    fig.savefig(OUT / "fig_r_offset.png", dpi=200)
    print("[written]", OUT / "fig_r_offset.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
