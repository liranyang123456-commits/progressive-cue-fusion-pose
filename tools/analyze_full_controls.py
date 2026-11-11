#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-budget 2x2 controls (retained seeds x beam width) on both the real BOP frames (both disjoint samples,
104 + 104 = 208 frames) and the synthetic Stanford meshes (the 31 views whose lowest-loss Stage-I proposal is
outside 10 deg, plus the first 5 views per mesh), together with the seeded vs unseeded BOP comparison and the
104-frame seeded 1/8-budget BOP evaluation.

  python tools/analyze_full_controls.py
Writes benchmark_results/full_controls_summary.json and manuscript/.../fullctrl_macros.tex.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import statistics as st
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp_bop", Path(__file__).resolve().parent / "export_bop_results.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)
eb.BOP_ROOT = Path(os.environ.get("BOP_ROOT", "data/bop"))
MAN = ROOT / "manuscript" / "computational_visual_media"
RES = ROOT if (ROOT / "bop6dof_results").exists() else ROOT / "results"
RUNS = ROOT / "benchmark_runs" if (ROOT / "benchmark_runs").exists() else ROOT / "results" / "benchmark_runs"
SECOND_DIR = "bop6dof_results_ext" if (RES / "bop6dof_results_ext").exists() else "bop6dof_results_second"

MESHES = [("bunny", 100, "Bunny"), ("dragon", 100, "Dragon"), ("armadillo", 50, "Armadillo"),
          ("happy", 50, "Happy Buddha"), ("lucy", 50, "Lucy")]
BOP_EXP = {"lmo": 40, "lm": 32, "tless": 32}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2.0 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n)


def ptxt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"={p:.3f}" if p < 0.01 else f"={p:.2f}"


def load_bop(base: Path, ds: str, fn: str) -> dict[str, dict]:
    p = base / ds / fn
    if not p.exists():
        return {}
    eb.RES = base if (base / ds / sorted(BOP_EXP)[0]).exists() or any((base / ds).glob("s*_im*_o*")) else RES / "bop6dof_results"
    # ensure gt_rot finds protocol.json even for archive/fast folders without frame subdirs
    if not any((base / ds).glob("s*_im*_o*")):
        eb.RES = RES / ("bop6dof_results_ext" if "ext" in base.name or "second" in base.name else "bop6dof_results")
    else:
        eb.RES = base
    recs = [r for r in eb.read_jsonl(p) if "e_R_deg" in r and "tag" in r]
    uniq = {r["tag"]: r for r in recs}
    vals = list(uniq.values())
    eb.add_sym_error(ds, vals)
    return {r["tag"]: r for r in vals}


def analyze_bop() -> dict:
    samples = [("A", RES / "bop6dof_results"), ("B", RES / SECOND_DIR)]
    variants = {"k8_b48": "results.jsonl", "k8_b1": "results_w1.jsonl",
                "k1_b48": "results_k1.jsonl", "k1_b1": "results_k1_w1.jsonl"}
    ok_by = {s: {v: [] for v in variants} for s in ("A", "B", "AB")}
    er_by = {s: {v: [] for v in variants} for s in ("A", "B", "AB")}
    tloc_by = {v: [] for v in variants}
    per_ds = {}
    complete = True

    for sname, sroot in samples:
        per_ds[sname] = {}
        for ds, disp in eb.DATASETS:
            tabs = {v: load_bop(sroot, ds, fn) for v, fn in variants.items()}
            tags = sorted(set.intersection(*(set(tabs[v]) for v in variants))) if all(tabs.values()) else []
            complete &= len(tags) == BOP_EXP[ds]
            per_ds[sname][ds] = {
                "n": len(tags),
                **{v: int(sum(tabs[v][t]["e_R_sym_deg"] <= 10.0 for t in tags)) for v in variants},
            }
            for v in variants:
                oks = [tabs[v][t]["e_R_sym_deg"] <= 10.0 for t in tags]
                ers = [float(tabs[v][t]["e_R_sym_deg"]) for t in tags]
                ok_by[sname][v] += oks
                ok_by["AB"][v] += oks
                er_by[sname][v] += ers
                er_by["AB"][v] += ers
                tloc_by[v] += [float(tabs[v][t]["t_local_s"]) for t in tags if tabs[v][t].get("t_local_s") is not None]

    def pair_stats(s: str, va: str, vb: str) -> dict:
        oa, ob = ok_by[s][va], ok_by[s][vb]
        ea, eb_ = er_by[s][va], er_by[s][vb]
        n = len(oa)
        only_a = int(sum(x and not y for x, y in zip(oa, ob)))
        only_b = int(sum(y and not x for x, y in zip(oa, ob)))
        p = mcnemar(only_a, only_b)
        diffs = [abs(x - y) for x, y in zip(ea, eb_)]
        return {
            "n": n, "ok_first": int(sum(oa)), "ok_second": int(sum(ob)),
            "sr_first": 100.0 * sum(oa) / max(1, n), "sr_second": 100.0 * sum(ob) / max(1, n),
            "only_first": only_a, "only_second": only_b, "p": p, "p_txt": ptxt(p),
            "identical_1e6": int(sum(d < 1e-6 for d in diffs)),
            "within_1deg": int(sum(d <= 1.0 for d in diffs)),
        }

    out = {"complete": complete, "per_dataset": per_ds, "samples": {}}
    for s in ("A", "B", "AB"):
        n = len(ok_by[s]["k8_b48"])
        out["samples"][s] = {
            "n": n,
            "ok": {v: int(sum(ok_by[s][v])) for v in variants},
            "sr": {v: round(100.0 * sum(ok_by[s][v]) / max(1, n), 2) for v in variants},
            "med_er_sym": {v: round(float(st.median(er_by[s][v])), 2) if er_by[s][v] else None for v in variants},
            "beam_vs_walk_k8": pair_stats(s, "k8_b48", "k8_b1"),
            "beam_vs_walk_k1": pair_stats(s, "k1_b48", "k1_b1"),
            "k8_vs_k1_beam": pair_stats(s, "k8_b48", "k1_b48"),
            "k8_vs_k1_walk": pair_stats(s, "k8_b1", "k1_b1"),
        }
    if tloc_by["k8_b48"] and tloc_by["k8_b1"]:
        out["t_local_median_s"] = {v: round(float(st.median(tloc_by[v])), 2) for v in variants if tloc_by[v]}

    # 3-frame scratch verification (without --reuse-stage1)
    chk = load_bop(RES / "bop6dof_results", "lmo", "results_scratchchk_w1.jsonl")
    reuse = load_bop(RES / "bop6dof_results", "lmo", "results_w1.jsonl")
    ctags = sorted(set(chk) & set(reuse))
    if ctags:
        d = [abs(float(chk[t]["e_R_deg"]) - float(reuse[t]["e_R_deg"])) for t in ctags]
        out["scratch_check_lmo"] = {
            "n": len(ctags), "identical_1e6": int(sum(x < 1e-6 for x in d)),
            "same_outcome": int(sum((chk[t]["e_R_sym_deg"] <= 10) == (reuse[t]["e_R_sym_deg"] <= 10) for t in ctags)),
            "max_diff_deg": round(float(max(d)), 4),
        }

    # Seeded vs archived unseeded full-budget BOP (when archive folder is present)
    bak = ROOT / "bop6dof_unseeded_archive"
    if bak.exists():
        sv_u = {}
        for sname, sdir in (("A", "bop6dof_results"), ("B", "bop6dof_results_ext")):
            ok_s, ok_u = [], []
            for ds, _ in eb.DATASETS:
                ts = load_bop(RES / (SECOND_DIR if sname == "B" else "bop6dof_results"), ds, "results.jsonl")
                tu = load_bop(bak / sdir, ds, "results.jsonl")
                common = sorted(set(ts) & set(tu))
                ok_s += [ts[t]["e_R_sym_deg"] <= 10.0 for t in common]
                ok_u += [tu[t]["e_R_sym_deg"] <= 10.0 for t in common]
            b = int(sum(x and not y for x, y in zip(ok_s, ok_u)))
            c = int(sum(y and not x for x, y in zip(ok_s, ok_u)))
            sv_u[sname] = {"n": len(ok_s), "seeded_ok": int(sum(ok_s)), "unseeded_ok": int(sum(ok_u)),
                           "only_seeded": b, "only_unseeded": c, "p": mcnemar(b, c)}
        out["seeded_vs_unseeded_archive"] = sv_u

    # 104-frame seeded 1/8-budget vs seeded full-budget (Sample A)
    fast_dir = RES / "bop6dof_results_fast_seeded"
    if fast_dir.exists():
        ok_f, ok_full = [], []
        for ds, _ in eb.DATASETS:
            tf = load_bop(fast_dir, ds, "results_fast.jsonl")
            to = load_bop(RES / "bop6dof_results", ds, "results.jsonl")
            common = sorted(set(tf) & set(to))
            ok_f += [tf[t]["e_R_sym_deg"] <= 10.0 for t in common]
            ok_full += [to[t]["e_R_sym_deg"] <= 10.0 for t in common]
        if ok_f:
            b = int(sum(x and not y for x, y in zip(ok_f, ok_full)))
            c = int(sum(y and not x for x, y in zip(ok_f, ok_full)))
            out["seeded_fast_vs_full_A"] = {
                "n": len(ok_f), "fast_ok": int(sum(ok_f)), "full_ok": int(sum(ok_full)),
                "fast_sr": round(100.0 * sum(ok_f) / len(ok_f), 1),
                "full_sr": round(100.0 * sum(ok_full) / len(ok_full), 1),
                "only_fast": b, "only_full": c, "p": mcnemar(b, c), "p_txt": ptxt(mcnemar(b, c)),
            }
    return out


def synth_rows(ds: str, n: int, method: str) -> dict[str, dict]:
    p = RUNS / f"{ds}_{n}gt" / method / "pose_metrics_summary.csv"
    if not p.exists():
        return {}
    return {r["tag"]: r for r in csv.DictReader(p.open(encoding="utf-8-sig")) if r.get("tag", "").startswith("gt_")}


def analyze_synth() -> dict:
    variants = {"k16_b48": "ours_v10_clean", "k16_b1": "ours_v10_w1",
                "k1_b48": "ours_v10_k1", "k1_b1": "ours_v10_k1w1"}
    ok_t = {v: [] for v in variants}
    ok_flip = {v: [] for v in variants}
    er_t = {v: [] for v in variants}
    tloc = {v: [] for v in variants}
    ctl = {"n": 0, "clean_ok": 0, "w1_ok": 0, "arch_ok": 0, "clean_vs_arch_identical": 0, "clean_vs_arch_max_diff": 0.0}
    per_mesh = {}
    n_views_all, s1_ok_all, n_target_exp = 0, 0, 0
    complete = True

    for ds, n, disp in MESHES:
        s1 = synth_rows(ds, n, "ours_v10_stage1")
        arch = synth_rows(ds, n, "ours_v10")
        tabs = {v: synth_rows(ds, n, m) for v, m in variants.items()}
        target = sorted(t for t, r in s1.items() if float(r["rot_err_deg"]) > 10.0)
        n_views_all += len(s1)
        s1_ok_all += len(s1) - len(target)
        n_target_exp += len(target)
        common_t = [t for t in target if all(t in tabs[v] for v in variants)]
        complete &= len(common_t) == len(target)
        flips = [t for t in common_t if float(s1[t]["rot_err_deg"]) > 30.0]
        per_mesh[ds] = {
            "n_views": len(s1), "stage1_ok": len(s1) - len(target), "n_target": len(target), "n_run": len(common_t),
            **{v: int(sum(float(tabs[v][t]["rot_err_deg"]) <= 10.0 for t in common_t)) for v in variants},
        }
        for v in variants:
            ok_t[v] += [float(tabs[v][t]["rot_err_deg"]) <= 10.0 for t in common_t]
            ok_flip[v] += [float(tabs[v][t]["rot_err_deg"]) <= 10.0 for t in flips]
            er_t[v] += [float(tabs[v][t]["rot_err_deg"]) for t in common_t]
            tloc[v] += [float(tabs[v][t]["t_local_s"]) for t in common_t if tabs[v][t].get("t_local_s")]

        # clean vs w1 vs archived ours_v10 on all common views (target | first5)
        all_cw = sorted(set(tabs["k16_b48"]) & set(tabs["k16_b1"]) & set(arch))
        for t in all_cw:
            ec = float(tabs["k16_b48"][t]["rot_err_deg"])
            ew = float(tabs["k16_b1"][t]["rot_err_deg"])
            ea = float(arch[t]["rot_err_deg"])
            ctl["n"] += 1
            ctl["clean_ok"] += int(ec <= 10.0)
            ctl["w1_ok"] += int(ew <= 10.0)
            ctl["arch_ok"] += int(ea <= 10.0)
            ctl["clean_vs_arch_identical"] += int(abs(ec - ea) < 1e-9)
            ctl["clean_vs_arch_max_diff"] = max(ctl["clean_vs_arch_max_diff"], abs(ec - ea))

    def pair(va: str, vb: str) -> dict:
        oa, ob = ok_t[va], ok_t[vb]
        b = int(sum(x and not y for x, y in zip(oa, ob)))
        c = int(sum(y and not x for x, y in zip(oa, ob)))
        p = mcnemar(b, c)
        return {"only_first": b, "only_second": c, "p": p, "p_txt": ptxt(p)}

    nt = len(ok_t["k16_b48"])
    nf = len(ok_flip["k16_b48"])
    return {
        "complete": complete and nt == n_target_exp,
        "n_views": n_views_all, "stage1_ok": s1_ok_all, "n_target": nt, "n_flip": nf,
        "ok": {v: int(sum(ok_t[v])) for v in variants},
        "ok_flip": {v: int(sum(ok_flip[v])) for v in variants},
        "med_er": {v: round(float(st.median(er_t[v])), 2) if er_t[v] else None for v in variants},
        "t_local_median_s": {v: round(float(st.median(tloc[v])), 1) for v in variants if tloc[v]},
        "beam_vs_walk_k16": pair("k16_b48", "k16_b1"),
        "beam_vs_walk_k1": pair("k1_b48", "k1_b1"),
        "k16_vs_k1_beam": pair("k16_b48", "k1_b48"),
        "k16_vs_k1_walk": pair("k16_b1", "k1_b1"),
        "clean_vs_archived_all_common": ctl,
        "per_mesh": per_mesh,
    }


def main() -> int:
    bop = analyze_bop()
    syn = analyze_synth()
    summary = {"bop_full_2x2": bop, "synth_full_2x2": syn}
    res_dir = RES / "benchmark_results" if (RES / "benchmark_results").exists() else ROOT / "benchmark_results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "full_controls_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("BOP FULL 2x2:", json.dumps({k: v for k, v in bop.items() if k != "per_dataset"}))
    print("SYNTH FULL 2x2:", json.dumps({k: v for k, v in syn.items() if k != "per_mesh"}))

    if not MAN.exists():
        return 0
    if not (bop["complete"] and syn["complete"]):
        print(f"[macros] not written yet: bop complete={bop['complete']}, synth complete={syn['complete']}")
        return 0

    ab = bop["samples"]["AB"]
    sa, sb = bop["samples"]["A"], bop["samples"]["B"]
    bw_bop = ab["beam_vs_walk_k8"]
    kk_bop = ab["k8_vs_k1_beam"]
    kkw_bop = ab["k8_vs_k1_walk"]
    bw_syn = syn["beam_vs_walk_k16"]
    kk_syn = syn["k16_vs_k1_beam"]
    kkw_syn = syn["k16_vs_k1_walk"]

    L = [
        "% Auto-generated by tools/analyze_full_controls.py",
        f"\\providecommand{{\\FullSynthTargetN}}{{{syn['n_target']}}}",
        f"\\providecommand{{\\FullSynthFlipN}}{{{syn['n_flip']}}}",
        f"\\providecommand{{\\FullSynthKBeam}}{{{syn['ok']['k16_b48']}}}",
        f"\\providecommand{{\\FullSynthKWalk}}{{{syn['ok']['k16_b1']}}}",
        f"\\providecommand{{\\FullSynthOneBeam}}{{{syn['ok']['k1_b48']}}}",
        f"\\providecommand{{\\FullSynthOneWalk}}{{{syn['ok']['k1_b1']}}}",
        f"\\providecommand{{\\FullSynthFlipKBeam}}{{{syn['ok_flip']['k16_b48']}}}",
        f"\\providecommand{{\\FullSynthFlipKWalk}}{{{syn['ok_flip']['k16_b1']}}}",
        f"\\providecommand{{\\FullSynthFlipOneBeam}}{{{syn['ok_flip']['k1_b48']}}}",
        f"\\providecommand{{\\FullSynthFlipOneWalk}}{{{syn['ok_flip']['k1_b1']}}}",
        f"\\providecommand{{\\FullSynthBwOnlyBeam}}{{{bw_syn['only_first']}}}",
        f"\\providecommand{{\\FullSynthBwOnlyWalk}}{{{bw_syn['only_second']}}}",
        f"\\providecommand{{\\FullSynthBwP}}{{{bw_syn['p_txt']}}}",
        f"\\providecommand{{\\FullSynthKkOnlyK}}{{{kk_syn['only_first']}}}",
        f"\\providecommand{{\\FullSynthKkOnlyOne}}{{{kk_syn['only_second']}}}",
        f"\\providecommand{{\\FullSynthKkP}}{{{kk_syn['p_txt']}}}",
        f"\\providecommand{{\\FullSynthKkWalkP}}{{{kkw_syn['p_txt']}}}",
        f"\\providecommand{{\\FullBopN}}{{{ab['n']}}}",
        f"\\providecommand{{\\FullBopAN}}{{{sa['n']}}}",
        f"\\providecommand{{\\FullBopAKBeamOk}}{{{sa['ok']['k8_b48']}}}",
        f"\\providecommand{{\\FullBopAKWalkOk}}{{{sa['ok']['k8_b1']}}}",
        f"\\providecommand{{\\FullBopAOneBeamOk}}{{{sa['ok']['k1_b48']}}}",
        f"\\providecommand{{\\FullBopAOneWalkOk}}{{{sa['ok']['k1_b1']}}}",
        f"\\providecommand{{\\FullBopAKBeamSR}}{{{sa['sr']['k8_b48']:.0f}}}",
        f"\\providecommand{{\\FullBopAKWalkSR}}{{{sa['sr']['k8_b1']:.0f}}}",
        f"\\providecommand{{\\FullBopAOneBeamSR}}{{{sa['sr']['k1_b48']:.0f}}}",
        f"\\providecommand{{\\FullBopAOneWalkSR}}{{{sa['sr']['k1_b1']:.0f}}}",
        f"\\providecommand{{\\FullBopBKBeamSR}}{{{sb['sr']['k8_b48']:.0f}}}",
        f"\\providecommand{{\\FullBopBKWalkSR}}{{{sb['sr']['k8_b1']:.0f}}}",
        f"\\providecommand{{\\FullBopKBeamOk}}{{{ab['ok']['k8_b48']}}}",
        f"\\providecommand{{\\FullBopKWalkOk}}{{{ab['ok']['k8_b1']}}}",
        f"\\providecommand{{\\FullBopOneBeamOk}}{{{ab['ok']['k1_b48']}}}",
        f"\\providecommand{{\\FullBopOneWalkOk}}{{{ab['ok']['k1_b1']}}}",
        f"\\providecommand{{\\FullBopKBeamSR}}{{{ab['sr']['k8_b48']:.0f}}}",
        f"\\providecommand{{\\FullBopKWalkSR}}{{{ab['sr']['k8_b1']:.0f}}}",
        f"\\providecommand{{\\FullBopOneBeamSR}}{{{ab['sr']['k1_b48']:.0f}}}",
        f"\\providecommand{{\\FullBopOneWalkSR}}{{{ab['sr']['k1_b1']:.0f}}}",
        f"\\providecommand{{\\FullBopBwOnlyBeam}}{{{bw_bop['only_first']}}}",
        f"\\providecommand{{\\FullBopBwOnlyWalk}}{{{bw_bop['only_second']}}}",
        f"\\providecommand{{\\FullBopBwP}}{{{bw_bop['p_txt']}}}",
        f"\\providecommand{{\\FullBopKkOnlyK}}{{{kk_bop['only_first']}}}",
        f"\\providecommand{{\\FullBopKkOnlyOne}}{{{kk_bop['only_second']}}}",
        f"\\providecommand{{\\FullBopKkP}}{{{kk_bop['p_txt']}}}",
        f"\\providecommand{{\\FullBopKkWalkP}}{{{kkw_bop['p_txt']}}}",
    ]
    if "t_local_median_s" in bop:
        L += [
            f"\\providecommand{{\\FullBopBeamLocalT}}{{{bop['t_local_median_s']['k8_b48']:.0f}}}",
            f"\\providecommand{{\\FullBopWalkLocalT}}{{{bop['t_local_median_s']['k8_b1']:.0f}}}",
        ]
    (MAN / "fullctrl_macros.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[written] {MAN / 'fullctrl_macros.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
