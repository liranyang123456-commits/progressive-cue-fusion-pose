#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derive the geometry-only selection ablation ("Geom-only": RGB removed from VisualSelect)
from the artefacts of a completed ours_v10 run, without re-running the search.

For every view, ours_v10 stores all K beam-refined candidates together with their
geometric loss and the appearance-augmented selection score in
  <run>/gt_XX/gt_XX_rand_octant_local_multistart.json
VisualSelect picks argmin(score); the Geom-only variant picks argmin(loss_total), i.e.
exactly what VAL_SELECT_USE_VISUAL_SCORE=0 does in the main script (same candidates,
same beam losses). The pose metrics of the geometric pick are recomputed from the GT
pose with the same formulas as the benchmark (benchmark_geom.full_pose_metrics_row).

Usage:
  python tools/derive_geom_only.py --run-root benchmark_runs/bunny_100gt --src ours_v10 --dst ours_v10_geom
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_geom import full_pose_metrics_row  # noqa: E402

FIELDS = ["tag", "rot_err_deg", "cam_center_err", "view_dir_err_deg", "gt_dist", "gt_elev_deg", "gt_azim_deg",
          "est_dist", "est_elev_deg", "est_azim_deg", "dist_err", "elev_err_deg", "azim_err_deg",
          "t_total_s", "t_global_s", "t_dist_align_s", "t_local_s", "t_select_s", "t_refine_s"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--src", default="ours_v10")
    ap.add_argument("--dst", default="ours_v10_geom")
    args = ap.parse_args()
    run_root = Path(args.run_root)
    src = run_root / args.src
    dst = run_root / args.dst
    src_csv = src / "pose_metrics_summary.csv"
    if not src_csv.exists():
        print(f"[skip] {src_csv} missing")
        return 1
    rows_src = {r["tag"]: r for r in csv.DictReader(src_csv.open(encoding="utf-8-sig"))}
    dst.mkdir(parents=True, exist_ok=True)
    out_rows = []
    n_changed = 0
    for tag, r in rows_src.items():
        js = src / tag / f"{tag}_rand_octant_local_multistart.json"
        if not js.exists():
            print(f"[warn] {js.name} missing; keeping ours_v10 row")
            out_rows.append(r)
            continue
        # runs started before the summary_path fix have one JSONL line appended after the object: take the first object only
        d = json.JSONDecoder().raw_decode(js.read_text(encoding="utf-8"))[0]
        runs = d.get("runs", [])
        if not runs:
            out_rows.append(r)
            continue
        best = min(runs, key=lambda x: float(x["loss_total"]))
        base = {
            "tag": tag,
            "gt_dist": float(r["gt_dist"]), "gt_elev_deg": float(r["gt_elev_deg"]), "gt_azim_deg": float(r["gt_azim_deg"]),
            "est_dist": float(best["dist"]), "est_elev_deg": float(best["elev_deg"]), "est_azim_deg": float(best["azim_deg"]),
        }
        m = full_pose_metrics_row(base)
        row = {**r, **base, **m}
        # selection-stage time is not spent by the geometric pick
        row["t_select_s"] = "0.0000"
        if abs(float(best["dist"]) - float(r["est_dist"])) > 1e-9 or abs(float(best["elev_deg"]) - float(r["est_elev_deg"])) > 1e-9:
            n_changed += 1
        out_rows.append(row)
    with (dst / "pose_metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    meta = {"derived_from": args.src, "selector": "argmin beam geometric loss (VAL_SELECT_USE_VISUAL_SCORE=0)",
            "n_views": len(out_rows), "n_views_changed_vs_visual": n_changed}
    if (src / "bench_meta.json").exists():
        meta.update({k: v for k, v in json.loads((src / "bench_meta.json").read_text(encoding="utf-8")).items() if k.startswith("VAL_")})
        meta["VAL_SELECT_USE_VISUAL_SCORE"] = "0"
        meta["method"] = args.dst
    (dst / "bench_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] {dst}: {len(out_rows)} views, {n_changed} differ from VisualSelect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
