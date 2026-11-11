#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduced-budget controls on the synthetic benchmark: (1) the two auxiliary terms of the archived full-budget
runs, (2) the beam width.

The full-budget runs of the paper (ours_v10 in benchmark_runs/<mesh>_<N>gt) were produced with
  (a) an appearance term in the order of the stage-I seeds (VAL_GLOBAL_RGB_RERANK=1), and
  (b) on beam levels whose candidates have a mean elevation of at least 45 deg, depth/normal weights raised by
      2.5/1.5 and an appearance term (VAL_LOCAL_BEAM_HIGH_ELEV_AUX=1).
Both are off by default in the released code. This script compares, on the same views and the same GPU,
  ours_v10_fast_clean  reduced budget (1/8), both terms off   -- all views of the five meshes
  ours_v10_fast_aux    reduced budget (1/8), both terms on    -- first 10 views of every mesh and every view that
                                                              ours_v10_fast_clean leaves outside 10 deg
and sets both against the archived full-budget run with the terms (ours_v10).

Beam width: ours_v10_fast_w1 is ours_v10_fast_clean with a width-one walk in place of the width-24 beam (coarse and
fine pass; proposals, distance alignment and selection unchanged). It is run on every view in which the lowest-loss
proposal of ours_v10_fast_clean (ours_v10_fast_stage1, tools/derive_stage1_only.py) is outside 10 deg, i.e. the views
in which the later stages have to change the basin or the pose, and on the first 10 views of every mesh as an
unselected sample; beam and walk are paired per view (exact McNemar test).

Hypothesis retention: ours_v10_fast_k1 / ours_v10_fast_k1w1 refine only the lowest-loss seed (the K seeds are still
drawn; VAL_LOCAL_BEST_SEED_ONLY=1) with the beam / the width-one walk, on the same target views, which gives a 2 x 2
design (seeds refined: K or 1; search within a seed: beam or walk).

Writes benchmark_results/aux_control_summary.json and manuscript/.../aux_macros.tex.

  python tools/analyze_aux_control.py            # writes the macros once all views have finished
  python tools/analyze_aux_control.py --partial  # also writes them from an unfinished run (drafting only)
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "benchmark_runs"
if not RUNS.exists():
    RUNS = ROOT / "results" / "benchmark_runs"
MAN = ROOT / "manuscript" / "computational_visual_media"
MESHES = [("bunny", 100, "Bunny"), ("dragon", 100, "Dragon"), ("armadillo", 50, "Armadillo"), ("happy", 50, "Happy Buddha"), ("lucy", 50, "Lucy")]
THR = 10.0


def rows(ds: str, n: int, method: str) -> dict[str, dict]:
    p = RUNS / f"{ds}_{n}gt" / method / "pose_metrics_summary.csv"
    if not p.exists():
        return {}
    return {r["tag"]: r for r in csv.DictReader(p.open(encoding="utf-8-sig")) if r.get("tag", "").startswith("gt_")}


def e(r: dict) -> float:
    return float(r["rot_err_deg"])


def wrong_shell(r: dict) -> bool:
    return abs(float(r["est_dist"]) - float(r["gt_dist"])) > 0.25 * float(r["gt_dist"])


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def p_txt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"={p:.3f}" if p < 0.01 else f"={p:.2f}"


def beam_width(out: dict) -> dict:
    """Width-one walk against the width-24 beam, and all K seeds against the lowest-loss seed alone, on the views whose
    lowest-loss proposal is outside 10 deg (target set); beam and walk also on the first 10 views of every mesh."""
    W = {"n_views": 0, "stage1_ok": 0, "n_target": 0, "n": 0, "beam_ok": 0, "walk_ok": 0, "beam_only": 0, "walk_only": 0,
         "flip_n": 0, "flip_beam_ok": 0, "flip_walk_ok": 0, "ctl_n": 0, "ctl_beam_ok": 0, "ctl_walk_ok": 0, "missing": 0,
         "ret_n": 0, "ret_k_beam": 0, "ret_k_walk": 0, "ret_1_beam": 0, "ret_1_walk": 0, "ret_missing": 0,
         "ret_flip_n": 0, "ret_flip_k_beam": 0, "ret_flip_k_walk": 0, "ret_flip_1_beam": 0, "ret_flip_1_walk": 0,
         "ret_near_n": 0, "ret_near_1_beam": 0, "ret_near_1_walk": 0,
         "ret_k_only_beam": 0, "ret_1_only_beam": 0, "ret_k_only_walk": 0, "ret_1_only_walk": 0}
    eb, ew, per, tlb, tlw = [], [], {}, [], []
    for ds, n, disp in MESHES:
        c, w, s1 = rows(ds, n, "ours_v10_fast_clean"), rows(ds, n, "ours_v10_fast_w1"), rows(ds, n, "ours_v10_fast_stage1")
        k1, k1w = rows(ds, n, "ours_v10_fast_k1"), rows(ds, n, "ours_v10_fast_k1w1")
        both = sorted(t for t in s1 if t in c)
        target = [t for t in both if e(s1[t]) > THR]
        tags = [t for t in target if t in w]
        ctl = [t for t in (f"gt_{i:02d}" for i in range(10)) if t in w and t in c]
        b_ok = [e(c[t]) <= THR for t in tags]
        w_ok = [e(w[t]) <= THR for t in tags]
        flips = [t for t in tags if e(s1[t]) > 30.0]
        ret = [t for t in tags if t in k1 and t in k1w]
        rflip = [t for t in ret if e(s1[t]) > 30.0]
        rnear = [t for t in ret if e(s1[t]) <= 30.0]
        ok = lambda tab, ts: sum(e(tab[t]) <= THR for t in ts)  # noqa: E731
        per[ds] = {"n_views": len(both), "stage1_ok": len(both) - len(target), "n_target": len(target), "n": len(tags),
                   "beam_ok": sum(b_ok), "walk_ok": sum(w_ok), "walk_failed": [t for t in tags if e(w[t]) > THR],
                   "ctl_n": len(ctl), "ctl_beam_ok": ok(c, ctl), "ctl_walk_ok": ok(w, ctl),
                   "ret_n": len(ret), "ret_k_beam": ok(c, ret), "ret_k_walk": ok(w, ret), "ret_1_beam": ok(k1, ret), "ret_1_walk": ok(k1w, ret),
                   "ret_flip_n": len(rflip), "ret_flip_k_beam": ok(c, rflip), "ret_flip_k_walk": ok(w, rflip),
                   "ret_flip_1_beam": ok(k1, rflip), "ret_flip_1_walk": ok(k1w, rflip),
                   "ret_near_n": len(rnear), "ret_near_1_beam": ok(k1, rnear), "ret_near_1_walk": ok(k1w, rnear)}
        W["n_views"] += len(both)
        W["stage1_ok"] += len(both) - len(target)
        W["n_target"] += len(target)
        W["missing"] += len(target) - len(tags) + (10 - len(ctl))
        W["ret_missing"] += len(target) - len(ret)
        W["n"] += len(tags)
        W["beam_ok"] += sum(b_ok)
        W["walk_ok"] += sum(w_ok)
        W["beam_only"] += sum(x and not y for x, y in zip(b_ok, w_ok))
        W["walk_only"] += sum(y and not x for x, y in zip(b_ok, w_ok))
        W["flip_n"] += len(flips)
        W["flip_beam_ok"] += ok(c, flips)
        W["flip_walk_ok"] += ok(w, flips)
        W["ret_k_only_beam"] += sum(e(c[t]) <= THR < e(k1[t]) for t in ret)
        W["ret_1_only_beam"] += sum(e(k1[t]) <= THR < e(c[t]) for t in ret)
        W["ret_k_only_walk"] += sum(e(w[t]) <= THR < e(k1w[t]) for t in ret)
        W["ret_1_only_walk"] += sum(e(k1w[t]) <= THR < e(w[t]) for t in ret)
        for k in ("ctl_n", "ctl_beam_ok", "ctl_walk_ok", "ret_n", "ret_k_beam", "ret_k_walk", "ret_1_beam", "ret_1_walk", "ret_flip_n",
                  "ret_flip_k_beam", "ret_flip_k_walk", "ret_flip_1_beam", "ret_flip_1_walk", "ret_near_n", "ret_near_1_beam", "ret_near_1_walk"):
            W[k] += per[ds][k]
        eb += [e(c[t]) for t in tags]
        ew += [e(w[t]) for t in tags]
        tlb += [float(c[t]["t_local_s"]) for t in tags if c[t].get("t_local_s")]
        tlw += [float(w[t]["t_local_s"]) for t in tags if w[t].get("t_local_s")]
        print(f"{disp:13s} lowest-loss proposal within 10 deg in {len(both) - len(target)}/{len(both)}; on the other {len(target)} "
              f"({len(tags)} run): walk {sum(w_ok)}, beam {sum(b_ok)} | best seed only ({len(ret)} run): beam {per[ds]['ret_1_beam']}, "
              f"walk {per[ds]['ret_1_walk']} | first 10: walk {per[ds]['ctl_walk_ok']}/{len(ctl)}, beam {per[ds]['ctl_beam_ok']}/{len(ctl)}")
    if W["n"]:
        W.update(beam_median=st.median(eb), walk_median=st.median(ew), p=mcnemar_exact(W["beam_only"], W["walk_only"]),
                 beam_sr=100.0 * W["beam_ok"] / W["n"], walk_sr=100.0 * W["walk_ok"] / W["n"], per_mesh=per)
        W["p_txt"] = p_txt(W["p"])
        if tlb and tlw:
            W.update(t_local_beam=st.median(tlb), t_local_walk=st.median(tlw))
    if W["ret_n"]:
        W["ret_p_beam"] = mcnemar_exact(W["ret_k_only_beam"], W["ret_1_only_beam"])
        W["ret_p_walk"] = mcnemar_exact(W["ret_k_only_walk"], W["ret_1_only_walk"])
        W["ret_p_beam_txt"], W["ret_p_walk_txt"] = p_txt(W["ret_p_beam"]), p_txt(W["ret_p_walk"])
    out["beam_width"] = W
    return W


def write_table(out: dict, W: dict) -> None:
    """Supplementary table: per-mesh outcome of the two reduced-budget controls."""
    L = ["% Auto-generated by tools/analyze_aux_control.py", "\\begin{table}[!htbp]", "\\centering", "\\footnotesize",
         "\\caption{Reduced-budget controls on the synthetic benchmark (an eighth of the proposal and beam budgets, one GPU). "
         "\\emph{Auxiliary terms}: views within $10^\\circ$ and median $e_R$ without the two auxiliary terms, against the "
         "full-budget runs of the paper, which contained them; on the paired views (first 10 of every mesh and every view "
         "that the run without the terms leaves outside $10^\\circ$) both variants ran at the reduced budget. "
         "\\emph{Search structure}: views whose lowest-loss proposal is within $10^\\circ$ (stage~I only), and, on the other "
         "views, those brought within $10^\\circ$ when all $K{=}8$ seeds or only the lowest-loss seed are refined, with the "
         "width-24 beam or a width-one walk.}",
         "\\label{tab:reduced_controls}", "\\setlength{\\tabcolsep}{3.5pt}",
         "\\resizebox{\\linewidth}{!}{%",
         "\\begin{tabular}{@{}l c cc cc ccc c c cc cc@{}}", "\\toprule",
         " & & \\multicolumn{2}{c}{without terms, $1/8$} & \\multicolumn{2}{c}{with terms, full} & "
         "\\multicolumn{3}{c}{paired views, $1/8$} & stage~I & & \\multicolumn{2}{c}{$K$ seeds} & \\multicolumn{2}{c}{best seed} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-10}\\cmidrule(lr){12-13}\\cmidrule(l){14-15}",
         "Mesh & $N$ & $\\le10^\\circ$ & med.\\ $e_R$ & $\\le10^\\circ$ & med.\\ $e_R$ & $n$ & without / with & identical & "
         "$\\le10^\\circ$ & $n$ & beam & walk & beam & walk \\\\", "\\midrule"]
    tot = {k: 0 for k in ("n", "c", "f", "np", "pc", "pa", "id", "s1", "nt", "b", "w", "b1", "w1")}
    for ds, n, disp in MESHES:
        m, pm = out["meshes"].get(ds, {}), W.get("per_mesh", {}).get(ds, {})
        if not m.get("n_clean"):
            continue
        L.append(f"{disp} & {m['n_clean']} & {m['clean_ok']} & {m['clean_median']:.2f} & {m['full_ok']} & {m['full_median']:.2f} & "
                 f"{m.get('n_pair', 0)} & {m.get('pair_clean_ok', '--')} / {m.get('pair_aux_ok', '--')} & {m.get('pair_identical', '--')} & "
                 f"{pm.get('stage1_ok', '--')} & {pm.get('ret_n', '--')} & {pm.get('ret_k_beam', '--')} & {pm.get('ret_k_walk', '--')} & "
                 f"{pm.get('ret_1_beam', '--')} & {pm.get('ret_1_walk', '--')} \\\\")
        for k, v in (("n", m["n_clean"]), ("c", m["clean_ok"]), ("f", m["full_ok"]), ("np", m.get("n_pair", 0)),
                     ("pc", m.get("pair_clean_ok", 0)), ("pa", m.get("pair_aux_ok", 0)), ("id", m.get("pair_identical", 0)),
                     ("s1", pm.get("stage1_ok", 0)), ("nt", pm.get("ret_n", 0)), ("b", pm.get("ret_k_beam", 0)), ("w", pm.get("ret_k_walk", 0)),
                     ("b1", pm.get("ret_1_beam", 0)), ("w1", pm.get("ret_1_walk", 0))):
            tot[k] += v
    P = out["pooled"]
    L += ["\\midrule",
          f"All & {tot['n']} & {tot['c']} & {P['clean_median']:.2f} & {tot['f']} & {P['full_median']:.2f} & {tot['np']} & "
          f"{tot['pc']} / {tot['pa']} & {tot['id']} & {tot['s1']} & {tot['nt']} & {tot['b']} & {tot['w']} & {tot['b1']} & {tot['w1']} \\\\",
          "\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (MAN / "tables" / "tab_reduced_controls.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    out: dict = {"meshes": {}}
    P = {"n": 0, "clean_ok": 0, "full_ok": 0, "clean_only": 0, "full_only": 0, "clean_wrong_shell": 0,
         "np": 0, "pc_ok": 0, "pa_ok": 0, "pc_only": 0, "pa_only": 0, "identical": 0}
    all_clean, all_full, pair_c, pair_a, pair_d, t_c, t_a = [], [], [], [], [], [], []
    for ds, n, disp in MESHES:
        c, a, f = rows(ds, n, "ours_v10_fast_clean"), rows(ds, n, "ours_v10_fast_aux"), rows(ds, n, "ours_v10")
        tags = sorted(t for t in c if t in f)
        m = {"n_clean": len(tags), "n_expected": n}
        if tags:
            ec, ef = [e(c[t]) for t in tags], [e(f[t]) for t in tags]
            m.update(clean_ok=sum(x <= THR for x in ec), full_ok=sum(x <= THR for x in ef),
                     clean_only=sum(x <= THR < y for x, y in zip(ec, ef)), full_only=sum(y <= THR < x for x, y in zip(ec, ef)),
                     clean_median=st.median(ec), full_median=st.median(ef), clean_mean=st.mean(ec), full_mean=st.mean(ef),
                     clean_max=max(ec), clean_wrong_shell=sum(wrong_shell(c[t]) for t in tags),
                     clean_failed=[t for t in tags if e(c[t]) > THR])
            all_clean += ec
            all_full += ef
            for k in ("clean_ok", "full_ok", "clean_only", "full_only", "clean_wrong_shell"):
                P[k] += m[k]
            P["n"] += len(tags)
            t_c += [float(c[t]["t_total_s"]) for t in tags if c[t].get("t_total_s")]
        ptags = sorted(t for t in a if t in c)
        m["n_pair"] = len(ptags)
        if ptags:
            pc, pa = [e(c[t]) for t in ptags], [e(a[t]) for t in ptags]
            d = [abs(x - y) for x, y in zip(pc, pa)]
            m.update(pair_clean_ok=sum(x <= THR for x in pc), pair_aux_ok=sum(x <= THR for x in pa),
                     pair_clean_only=sum(x <= THR < y for x, y in zip(pc, pa)), pair_aux_only=sum(y <= THR < x for x, y in zip(pc, pa)),
                     pair_identical=sum(x < 1e-9 for x in d), pair_max_diff=max(d), pair_median_diff=st.median(d),
                     pair_clean_median=st.median(pc), pair_aux_median=st.median(pa))
            pair_c += pc
            pair_a += pa
            pair_d += d
            P["np"] += len(ptags)
            P["pc_ok"] += m["pair_clean_ok"]
            P["pa_ok"] += m["pair_aux_ok"]
            P["pc_only"] += m["pair_clean_only"]
            P["pa_only"] += m["pair_aux_only"]
            P["identical"] += m["pair_identical"]
            t_a += [float(a[t]["t_total_s"]) for t in ptags if a[t].get("t_total_s")]
        out["meshes"][ds] = m
        print(f"{disp:13s} clean {m.get('clean_ok', '-')}/{m['n_clean']} (full budget with terms: {m.get('full_ok', '-')}/{m['n_clean']}), "
              f"median e_R {m.get('clean_median', float('nan')):.2f} vs {m.get('full_median', float('nan')):.2f} | "
              f"paired {m['n_pair']}: clean {m.get('pair_clean_ok', '-')} / aux {m.get('pair_aux_ok', '-')}, identical {m.get('pair_identical', '-')}, "
              f"max |diff| {m.get('pair_max_diff', float('nan')):.2f}")
    if P["n"]:
        P.update(clean_median=st.median(all_clean), full_median=st.median(all_full), clean_mean=st.mean(all_clean),
                 clean_sr=100.0 * P["clean_ok"] / P["n"], full_sr=100.0 * P["full_ok"] / P["n"])
    if P["np"]:
        srt = sorted(pair_d)
        P.update(pair_clean_median=st.median(pair_c), pair_aux_median=st.median(pair_a), pair_median_diff=st.median(pair_d),
                 pair_p95_diff=srt[int(0.95 * (len(srt) - 1))], pair_max_diff=max(pair_d),
                 pair_within_half_deg=sum(x <= 0.5 for x in pair_d))
    if t_c:
        P["t_clean_median"] = st.median(t_c)
    if t_a:
        P["t_aux_median"] = st.median(t_a)
    out["pooled"] = P
    W = beam_width(out)
    res_dir = RUNS.parent / "benchmark_results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "aux_control_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("POOLED", json.dumps(P))
    print("BEAM WIDTH", json.dumps({k: v for k, v in W.items() if k != "per_mesh"}))
    n_all = sum(n for _, n, _ in MESHES)
    aux_missing = sum(1 for ds, n, _ in MESHES for t, r in rows(ds, n, "ours_v10_fast_clean").items()
                      if (e(r) > THR or t in {f"gt_{i:02d}" for i in range(10)}) and t not in rows(ds, n, "ours_v10_fast_aux"))
    complete = P["n"] == n_all and W["n_views"] == n_all and W["missing"] == 0 and W["ret_missing"] == 0 and aux_missing == 0
    if MAN.exists() and (complete or "--partial" in sys.argv) and P["np"]:
        L = ["% Auto-generated by tools/analyze_aux_control.py" if complete else
             "% DRAFT from unfinished runs (tools/analyze_aux_control.py --partial): not for submission",
             f"\\providecommand{{\\AuxN}}{{{P['n']}}}",
             f"\\providecommand{{\\AuxCleanOk}}{{{P['clean_ok']}}}",
             f"\\providecommand{{\\AuxCleanSR}}{{{P['clean_sr']:.1f}}}",
             f"\\providecommand{{\\AuxFullOk}}{{{P['full_ok']}}}",
             f"\\providecommand{{\\AuxCleanOnly}}{{{P['clean_only']}}}",
             f"\\providecommand{{\\AuxFullOnly}}{{{P['full_only']}}}",
             f"\\providecommand{{\\AuxCleanMedian}}{{{P['clean_median']:.2f}}}",
             f"\\providecommand{{\\AuxFullMedian}}{{{P['full_median']:.2f}}}",
             f"\\providecommand{{\\AuxCleanWrongShell}}{{{P['clean_wrong_shell']}}}",
             f"\\providecommand{{\\AuxPairN}}{{{P['np']}}}",
             f"\\providecommand{{\\AuxPairCleanOk}}{{{P['pc_ok']}}}",
             f"\\providecommand{{\\AuxPairAuxOk}}{{{P['pa_ok']}}}",
             f"\\providecommand{{\\AuxPairCleanOnly}}{{{P['pc_only']}}}",
             f"\\providecommand{{\\AuxPairAuxOnly}}{{{P['pa_only']}}}",
             f"\\providecommand{{\\AuxPairIdentical}}{{{P['identical']}}}",
             f"\\providecommand{{\\AuxPairCleanMedian}}{{{P['pair_clean_median']:.2f}}}",
             f"\\providecommand{{\\AuxPairAuxMedian}}{{{P['pair_aux_median']:.2f}}}",
             f"\\providecommand{{\\AuxPairMedianDiff}}{{{P['pair_median_diff']:.2f}}}",
             f"\\providecommand{{\\AuxPairMaxDiff}}{{{P['pair_max_diff']:.1f}}}",
             f"\\providecommand{{\\AuxPairWithinHalf}}{{{P['pair_within_half_deg']}}}",
             f"\\providecommand{{\\AuxTimeClean}}{{{P.get('t_clean_median', float('nan')):.0f}}}",
             f"\\providecommand{{\\AuxTimeAux}}{{{P.get('t_aux_median', float('nan')):.0f}}}"]
        if W.get("n"):
            L += [f"\\providecommand{{\\WoneStageOneOk}}{{{W['stage1_ok']}}}",
                  f"\\providecommand{{\\WoneViews}}{{{W['n_views']}}}",
                  f"\\providecommand{{\\WoneFlipN}}{{{W['flip_n']}}}",
                  f"\\providecommand{{\\WoneFlipBeamOk}}{{{W['flip_beam_ok']}}}",
                  f"\\providecommand{{\\WoneFlipWalkOk}}{{{W['flip_walk_ok']}}}",
                  f"\\providecommand{{\\WoneCtlN}}{{{W['ctl_n']}}}",
                  f"\\providecommand{{\\WoneCtlBeamOk}}{{{W['ctl_beam_ok']}}}",
                  f"\\providecommand{{\\WoneCtlWalkOk}}{{{W['ctl_walk_ok']}}}",
                  f"\\providecommand{{\\WoneN}}{{{W['n']}}}",
                  f"\\providecommand{{\\WoneBeamOk}}{{{W['beam_ok']}}}",
                  f"\\providecommand{{\\WoneWalkOk}}{{{W['walk_ok']}}}",
                  f"\\providecommand{{\\WoneBeamSR}}{{{W['beam_sr']:.1f}}}",
                  f"\\providecommand{{\\WoneWalkSR}}{{{W['walk_sr']:.1f}}}",
                  f"\\providecommand{{\\WoneBeamOnly}}{{{W['beam_only']}}}",
                  f"\\providecommand{{\\WoneWalkOnly}}{{{W['walk_only']}}}",
                  f"\\providecommand{{\\WoneBeamMedian}}{{{W['beam_median']:.2f}}}",
                  f"\\providecommand{{\\WoneWalkMedian}}{{{W['walk_median']:.2f}}}",
                  f"\\providecommand{{\\WoneP}}{{{W['p_txt']}}}",
                  f"\\providecommand{{\\WoneBeamLocalT}}{{{W.get('t_local_beam', float('nan')):.0f}}}",
                  f"\\providecommand{{\\WoneWalkLocalT}}{{{W.get('t_local_walk', float('nan')):.0f}}}"]
        if W.get("ret_n"):
            L += [f"\\providecommand{{\\RetN}}{{{W['ret_n']}}}",
                  f"\\providecommand{{\\RetKBeam}}{{{W['ret_k_beam']}}}",
                  f"\\providecommand{{\\RetKWalk}}{{{W['ret_k_walk']}}}",
                  f"\\providecommand{{\\RetOneBeam}}{{{W['ret_1_beam']}}}",
                  f"\\providecommand{{\\RetOneWalk}}{{{W['ret_1_walk']}}}",
                  f"\\providecommand{{\\RetFlipN}}{{{W['ret_flip_n']}}}",
                  f"\\providecommand{{\\RetFlipKBeam}}{{{W['ret_flip_k_beam']}}}",
                  f"\\providecommand{{\\RetFlipKWalk}}{{{W['ret_flip_k_walk']}}}",
                  f"\\providecommand{{\\RetFlipOneBeam}}{{{W['ret_flip_1_beam']}}}",
                  f"\\providecommand{{\\RetFlipOneWalk}}{{{W['ret_flip_1_walk']}}}",
                  f"\\providecommand{{\\RetNearN}}{{{W['ret_near_n']}}}",
                  f"\\providecommand{{\\RetNearOneBeam}}{{{W['ret_near_1_beam']}}}",
                  f"\\providecommand{{\\RetNearOneWalk}}{{{W['ret_near_1_walk']}}}",
                  f"\\providecommand{{\\RetKOnlyBeam}}{{{W['ret_k_only_beam']}}}",
                  f"\\providecommand{{\\RetOneOnlyBeam}}{{{W['ret_1_only_beam']}}}",
                  f"\\providecommand{{\\RetKOnlyWalk}}{{{W['ret_k_only_walk']}}}",
                  f"\\providecommand{{\\RetOneOnlyWalk}}{{{W['ret_1_only_walk']}}}",
                  f"\\providecommand{{\\RetPBeam}}{{{W['ret_p_beam_txt']}}}",
                  f"\\providecommand{{\\RetPWalk}}{{{W['ret_p_walk_txt']}}}"]
        else:  # drafting only: placeholders keep the documents compilable
            L += [f"\\providecommand{{\\Wone{k}}}{{??}}" for k in ("StageOneOk", "Views", "FlipN", "FlipBeamOk", "FlipWalkOk", "CtlN",
                                                                  "CtlBeamOk", "CtlWalkOk", "N", "BeamOk", "WalkOk", "BeamSR", "WalkSR",
                                                                  "BeamOnly", "WalkOnly", "BeamMedian", "WalkMedian")]
            L.append("\\providecommand{\\WoneP}{=??}")
        if not W.get("ret_n"):
            L += [f"\\providecommand{{\\Ret{k}}}{{??}}" for k in ("N", "KBeam", "KWalk", "OneBeam", "OneWalk", "FlipN", "FlipKBeam", "FlipKWalk",
                                                                 "FlipOneBeam", "FlipOneWalk", "NearN", "NearOneBeam", "NearOneWalk")]
            L += ["\\providecommand{\\RetPBeam}{=??}", "\\providecommand{\\RetPWalk}{=??}"]
        (MAN / "aux_macros.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
        write_table(out, W)
        print("[macros] aux_macros.tex and tables/tab_reduced_controls.tex written")
    elif not complete:
        print(f"[macros] not written yet: {P['n']} of {n_all} views without the terms, {aux_missing} paired views with them, "
              f"{W['missing']} width-one and {W['ret_missing']} single-seed views still to run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
