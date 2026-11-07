#!/usr/bin/env python3
"""
Per-frame complementarity of the initialisation-free estimators on the BOP frames: the proposed method,
WAPR's twelve-hypothesis estimator and pi3 + stages II-IV (symmetry-aware SR@10 deg, same footing as Table 4).

  python tools/analyze_complementarity.py
"""
from __future__ import annotations

import os

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp_bop", Path(__file__).resolve().parent / "export_bop_results.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)
eb.RES = ROOT / "bop6dof_results"
eb.BOP_ROOT = Path(os.environ.get("BOP_ROOT", "data/bop"))

FILES = {"ours": "results.jsonl", "wapr12": "baseline_wapr_full.jsonl", "pi3_ours": "results_init_pi3.jsonl",
         "ours_wapr": "baseline_wapr_from_ours.jsonl"}


def load(ds: str, fn: str) -> dict:
    uniq = {}
    for r in eb.read_jsonl(eb.RES / ds / fn):
        if "e_R_deg" in r and str(r.get("bin", "none")) in ("none", "ours", "pi3"):
            uniq[r["tag"]] = r
    rows = list(uniq.values())
    eb.add_sym_error(ds, rows)
    return {r["tag"]: r for r in rows}


def main() -> int:
    tot = {}
    for ds, name in eb.DATASETS:
        g = {k: load(ds, fn) for k, fn in FILES.items()}
        tags = sorted(g["ours"])
        ok = {k: {t: (t in g[k] and float(g[k][t]["e_R_sym_deg"]) <= 10) for t in tags} for k in g}
        cand_oracle = {t: any(float(c["e_R_deg"]) <= 10 for c in g["ours"][t].get("candidates", [])) for t in tags}
        n = len(tags)
        c = lambda f: sum(1 for t in tags if f(t))
        res = {
            "n": n,
            "ours": c(lambda t: ok["ours"][t]),
            "wapr12": c(lambda t: ok["wapr12"][t]),
            "pi3_ours": c(lambda t: ok["pi3_ours"][t]),
            "ours_wapr": c(lambda t: ok["ours_wapr"][t]),
            "ours_only_vs_wapr": c(lambda t: ok["ours"][t] and not ok["wapr12"][t]),
            "wapr_only_vs_ours": c(lambda t: ok["wapr12"][t] and not ok["ours"][t]),
            "union_ours_wapr": c(lambda t: ok["ours"][t] or ok["wapr12"][t]),
            "union_ours_pi3": c(lambda t: ok["ours"][t] or ok["pi3_ours"][t]),
            "cand_oracle_raw": c(lambda t: cand_oracle[t]),
        }
        tot[name] = res
        print(f"\n=== {name} (N={n}) ===")
        for k, v in res.items():
            if k != "n":
                print(f"  {k:22s} {v:3d}  ({100.0 * v / n:5.1f}%)")
        fails = [t for t in tags if not ok["wapr12"][t]]
        print("  WAPR-12 failures:", ", ".join(f"{t}(ours {float(g['ours'][t]['e_R_sym_deg']):.0f}deg)" for t in fails))
    N = sum(v["n"] for v in tot.values())
    print(f"\n=== pooled (N={N}) ===")
    for k in next(iter(tot.values())):
        if k != "n":
            s = sum(v[k] for v in tot.values())
            print(f"  {k:22s} {s:3d}  ({100.0 * s / N:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
