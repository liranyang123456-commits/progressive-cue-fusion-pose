"""Paired comparison of the occlusion-aware silhouette term against the default configuration on the BOP frames
(initialisation-free method, stages I-IV): SR@10 deg with the symmetry-aware rotation error, ADD(-S) recall,
AR_MSSD, and an exact McNemar test on the frames solved by only one of the two configurations.
Usage: python tools/compare_occlusion_aware.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_bop_results as E  # noqa: E402


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def variant(ds: str, fn: str) -> list[dict]:
    recs = [r for r in E.read_jsonl(E.RES / ds / fn) if "e_R_deg" in r]
    E.add_sym_error(ds, recs)
    return recs


def main() -> int:
    for fn, label in (("results_occ.jsonl", "occlusion-aware, margin r/4"), ("results_occ_m0.jsonl", "diagnostic: margin 0")):
        tb = tc = tn = 0
        print(f"== {label}")
        for ds in ("lmo", "lm", "tless"):
            base = {r["tag"]: r for r in E.load_all(ds).get("ours_full", [])}
            pair = [(base[r["tag"]], r) for r in variant(ds, fn) if r["tag"] in base]
            if not pair:
                continue
            sb, so = E.summarize([x for x, _ in pair]), E.summarize([y for _, y in pair])
            b = sum(1 for x, y in pair if y["e_R_sym_deg"] <= 10 < x["e_R_sym_deg"])
            c = sum(1 for x, y in pair if x["e_R_sym_deg"] <= 10 < y["e_R_sym_deg"])
            tb, tc, tn = tb + b, tc + c, tn + len(pair)
            print(f"{ds:5s} n={len(pair):2d}  SR10sym {100*sb['sr10_sym']:5.1f} -> {100*so['sr10_sym']:5.1f}  ADD {100*sb['add01']:5.1f} -> {100*so['add01']:5.1f}  "
                  f"ARmssd {sb['ar_mssd']:.2f} -> {so['ar_mssd']:.2f}  medERsym {sb['eRsym_median']:5.1f} -> {so['eRsym_median']:5.1f}  "
                  f"only-variant {b} only-default {c}  McNemar p={mcnemar_exact(b, c):.3f}")
        print(f"pooled n={tn}: only-variant {tb}, only-default {tc}, McNemar p={mcnemar_exact(tb, tc):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
