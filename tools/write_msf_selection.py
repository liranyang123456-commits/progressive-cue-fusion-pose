#!/usr/bin/env python3
"""
Write the per-frame selection over all hypothesis sources (proposed search, four feed-forward initialisations and the
re-seeded rotations, lowest final geometric loss, no ground truth) to <root>/<ds>/results_msf_select.jsonl, so that the
refiners can start from it (refine-from-ours with --ours-file results_msf_select.jsonl).

  python tools/write_msf_selection.py
"""
from __future__ import annotations

import os

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp_bop", Path(__file__).resolve().parent / "export_bop_results.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)
eb.BOP_ROOT = Path(os.environ.get("BOP_ROOT", "data/bop"))


def main() -> int:
    for root in (ROOT / "bop6dof_results", ROOT / "bop6dof_results_ext"):
        eb.RES = root
        for ds, _ in eb.DATASETS:
            g = eb.load_all(ds)
            sel = g.get("ours_msf") or g.get("ours_ms") or []
            out = root / ds / "results_msf_select.jsonl"
            out.write_text("".join(json.dumps(r) + "\n" for r in sel), encoding="utf-8")
            ok = sum(float(r["e_R_sym_deg"]) <= 10 for r in sel)
            n_re = sum(1 for r in sel if r.get("n_reseed"))
            print(f"{root.name}/{ds}: {len(sel)} frames ({n_re} with re-seeded hypotheses), SR@10 sym {100 * ok / max(1, len(sel)):.1f}%  -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
