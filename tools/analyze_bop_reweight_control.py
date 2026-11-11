#!/usr/bin/env python3
"""
Controls on the BOP frames at the reduced budget (bop_full6dof_eval.py --budget 1/8): beam re-weighting and the
dependence on the proposal draw.

The BOP runs of the paper were produced with the depth / normal weights of stage III raised on beam levels whose
candidates have a mean elevation of at least 45 deg (no appearance term enters any stage on sensor frames), and with
the proposals of stage I drawn from the default torch generator, which that code never seeded: every run drew other
proposals. The released code scores every beam candidate with the same weights and seeds the generators per frame
(--seed and the frame tag); --high-elev-weights and --no-frame-seed restore the archived behaviour.

On every third frame of Table 4 (--frame-stride 3: an object-balanced subset of 36 of the 104 frames) this script
compares
  bop6dof_results_fast_seeded/<ds>/results_fast.jsonl       same weights, seeded (released default)
  bop6dof_results_fast_seeded_rw/<ds>/results_fast.jsonl    --high-elev-weights, seeded: the same proposals
  bop6dof_results_fast_clean/<ds>/results_fast.jsonl        same weights, --no-frame-seed: another proposal draw
  bop6dof_results_fast/<ds>/results_fast.jsonl              archived reduced-budget run of the paper (re-weighted, unseeded)
(symmetry-aware SR@10 deg, exact McNemar tests) and checks on the first LM-O frames that a seeded run reproduces
itself (bop6dof_results_fast_seeded_repeat/lmo).

  python tools/analyze_bop_reweight_control.py
Writes benchmark_results/bop_reweight_control_summary.json and manuscript/.../bopaux_macros.tex.
"""
from __future__ import annotations

import os

import importlib.util
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp_bop", Path(__file__).resolve().parent / "export_bop_results.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)
eb.BOP_ROOT = Path(os.environ.get("BOP_ROOT", "data/bop"))
MAN = ROOT / "manuscript" / "computational_visual_media"
RES = ROOT if (ROOT / "bop6dof_results_fast").exists() else ROOT / "results"   # working tree / release layout
STRIDE = 3
N_EXPECTED = {"lmo": 14, "lm": 11, "tless": 11}  # every third of the 40 / 32 / 32 exported frames
RUNS = {"same": ("bop6dof_results_fast_seeded", "results_fast.jsonl"), "rew": ("bop6dof_results_fast_seeded_rw", "results_fast.jsonl"),
        "draw2": ("bop6dof_results_fast_clean", "results_fast.jsonl"), "arch": ("bop6dof_results_fast", "results_fast.jsonl"),
        "repeat": ("bop6dof_results_fast_seeded_repeat", "results_fast.jsonl")}


def load(key: str, ds: str) -> dict:
    root, name = RUNS[key]
    fn = RES / root / ds / name
    if not fn.exists():
        return {}
    eb.RES = RES / root
    rows = [r for r in eb.read_jsonl(fn) if "e_R_deg" in r]
    eb.add_sym_error(ds, rows)
    return {r["tag"]: r for r in rows}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def ptxt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"={p:.2f}"


def main() -> int:
    # a seeded run must reproduce itself: first LM-O frames, run twice
    a, b = load("same", "lmo"), load("repeat", "lmo")
    RUNS.pop("repeat")
    common = sorted(set(a) & set(b))
    rep_chk = {"n": len(common), "identical": int(sum(abs(float(a[t]["e_R_deg"]) - float(b[t]["e_R_deg"])) < 1e-9 for t in common))}
    ok = {k: [] for k in RUNS}          # outcome per frame, frames common to all runs
    est = {k: [] for k in RUNS}
    per, complete = {}, True
    for ds, name in eb.DATASETS:
        tabs = {k: load(k, ds) for k in RUNS}
        subset = sorted(tabs["arch"])[::STRIDE]
        tags = [t for t in subset if all(t in tabs[k] for k in RUNS)]
        complete &= len(tags) == N_EXPECTED.get(ds, -1)
        per[ds] = {"n": len(tags), **{k: int(sum(tabs[k][t]["e_R_sym_deg"] <= 10 for t in tags)) for k in RUNS}}
        for k in RUNS:
            ok[k] += [tabs[k][t]["e_R_sym_deg"] <= 10 for t in tags]
            est[k] += [float(tabs[k][t]["e_R_deg"]) for t in tags]
        print(f"{name:7s} N={len(tags):3d} of {len(subset)}: " + "  ".join(f"{k} {per[ds][k]}" for k in RUNS)
              + "  | finished: " + " ".join(f"{k} {sum(t in tabs[k] for t in subset)}" for k in RUNS))
    n = len(ok["same"])
    out: dict = {"datasets": per, "n": n}

    def pair(a: str, b: str) -> dict:
        only_a = sum(x and not y for x, y in zip(ok[a], ok[b]))
        only_b = sum(y and not x for x, y in zip(ok[a], ok[b]))
        d = [abs(x - y) for x, y in zip(est[a], est[b])]
        return {"only_first": only_a, "only_second": only_b, "p": mcnemar(only_a, only_b), "same_outcome": n - only_a - only_b,
                "identical": int(sum(x < 1e-6 for x in d)), "within_1deg": int(sum(x <= 1.0 for x in d))}

    out["ok"] = {k: int(sum(v)) for k, v in ok.items()}
    out["same_vs_reweighted"] = pair("same", "rew")
    out["seeded_repeat_lmo"] = rep_chk
    out["two_draws_same_weights"] = pair("same", "draw2")
    out["reweighted_vs_archived"] = pair("rew", "arch")
    (RES / "benchmark_results").mkdir(exist_ok=True)
    (RES / "benchmark_results" / "bop_reweight_control_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k in ("ok", "same_vs_reweighted", "two_draws_same_weights", "reweighted_vs_archived", "seeded_repeat_lmo"):
        if k in out:
            print(k, json.dumps(out[k]))
    if not MAN.exists():
        return 0
    if not (complete and n and rep_chk["n"]):
        print(f"[macros] not written: {n} of {sum(N_EXPECTED.values())} frames finished in all runs, {rep_chk['n']} repeated")
        return 0
    sr = lambda k: f"{100 * out['ok'][k] / n:.0f}"  # noqa: E731
    rw, dr, ar = out["same_vs_reweighted"], out["two_draws_same_weights"], out["reweighted_vs_archived"]
    L = ["% Auto-generated by tools/analyze_bop_reweight_control.py",
         f"\\providecommand{{\\BopAuxN}}{{{n}}}",
         f"\\providecommand{{\\BopAuxSameOk}}{{{out['ok']['same']}}}", f"\\providecommand{{\\BopAuxSameSR}}{{{sr('same')}}}",
         f"\\providecommand{{\\BopAuxReweightOk}}{{{out['ok']['rew']}}}", f"\\providecommand{{\\BopAuxReweightSR}}{{{sr('rew')}}}",
         f"\\providecommand{{\\BopAuxOnlySame}}{{{rw['only_first']}}}", f"\\providecommand{{\\BopAuxOnlyReweight}}{{{rw['only_second']}}}",
         f"\\providecommand{{\\BopAuxP}}{{{ptxt(rw['p'])}}}", f"\\providecommand{{\\BopAuxIdentical}}{{{rw['identical']}}}",
         f"\\providecommand{{\\BopDrawOk}}{{{out['ok']['draw2']}}}", f"\\providecommand{{\\BopDrawSR}}{{{sr('draw2')}}}",
         f"\\providecommand{{\\BopDrawSameOutcome}}{{{dr['same_outcome']}}}", f"\\providecommand{{\\BopDrawOnlyFirst}}{{{dr['only_first']}}}",
         f"\\providecommand{{\\BopDrawOnlySecond}}{{{dr['only_second']}}}", f"\\providecommand{{\\BopDrawIdentical}}{{{dr['identical']}}}",
         f"\\providecommand{{\\BopArchOk}}{{{out['ok']['arch']}}}", f"\\providecommand{{\\BopArchSR}}{{{sr('arch')}}}",
         f"\\providecommand{{\\BopArchSameOutcome}}{{{ar['same_outcome']}}}", f"\\providecommand{{\\BopArchIdentical}}{{{ar['identical']}}}"]
    L += [f"\\providecommand{{\\BopRepN}}{{{rep_chk['n']}}}", f"\\providecommand{{\\BopRepIdentical}}{{{rep_chk['identical']}}}",
          f"\\providecommand{{\\BopAuxWithinOne}}{{{rw['within_1deg']}}}"]
    (MAN / "bopaux_macros.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[written] {MAN / 'bopaux_macros.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
