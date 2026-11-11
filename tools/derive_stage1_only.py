#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derive the "stage I only" ablation (lowest-loss proposal of the proposed run, no distance alignment, no beam,
no selection) from the artefacts of a completed ours_v10 run, without re-running the search.

For every view, ours_v10 stores the K seeds that stage I hands to the later stages together with their
geometric loss in
  <run>/gt_XX/seed_00/gt_XX_rand_octant_best.json   ("topk_seeds")
The ablation takes the seed with the lowest loss as the estimate, so that it shares the proposals, the loss and
the seeds of the proposed run exactly. The pose metrics are recomputed from the ground-truth pose with the same
formulas as the benchmark (benchmark_geom.full_pose_metrics_row); the time is that of stage I.

Usage:
  python tools/derive_stage1_only.py --run-root benchmark_runs/bunny_100gt [--src ours_v10 --dst ours_v10_stage1]
  python tools/derive_stage1_only.py --all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_geom import full_pose_metrics_row  # noqa: E402

FIELDS = ["tag", "rot_err_deg", "cam_center_err", "view_dir_err_deg", "gt_dist", "gt_elev_deg", "gt_azim_deg",
          "est_dist", "est_elev_deg", "est_azim_deg", "dist_err", "elev_err_deg", "azim_err_deg",
          "t_total_s", "t_global_s", "t_dist_align_s", "t_local_s", "t_select_s", "t_refine_s"]
ALL = ["bunny_100gt", "dragon_100gt", "armadillo_50gt", "happy_50gt", "lucy_50gt"]


def derive(run_root: Path, src_name: str, dst_name: str) -> int:
    src, dst = run_root / src_name, run_root / dst_name
    src_csv = src / "pose_metrics_summary.csv"
    if not src_csv.exists():
        print(f"[skip] {src_csv} missing")
        return 1
    rows_src = {r["tag"]: r for r in csv.DictReader(src_csv.open(encoding="utf-8-sig"))}
    out_rows, missing = [], 0
    for tag, r in rows_src.items():
        js = src / tag / "seed_00" / f"{tag}_rand_octant_best.json"
        if not js.exists():
            missing += 1
            continue
        seeds = json.JSONDecoder().raw_decode(js.read_text(encoding="utf-8"))[0].get("topk_seeds", [])
        if not seeds:
            missing += 1
            continue
        best = min(seeds, key=lambda x: float(x["loss_total"]))
        base = {"tag": tag,
                "gt_dist": float(r["gt_dist"]), "gt_elev_deg": float(r["gt_elev_deg"]), "gt_azim_deg": float(r["gt_azim_deg"]),
                "est_dist": float(best["dist"]), "est_elev_deg": float(best["elev_deg"]), "est_azim_deg": float(best["azim_deg"])}
        row = {**r, **base, **full_pose_metrics_row(base)}
        t_global = r.get("t_global_s", "")
        row.update({"t_total_s": t_global, "t_dist_align_s": "0.0000", "t_local_s": "0.0000", "t_select_s": "0.0000", "t_refine_s": "0.0000"})
        out_rows.append(row)
    if missing:
        print(f"[warn] {run_root.name}: stage-I seed list missing for {missing} of {len(rows_src)} views")
    if not out_rows:
        return 1
    dst.mkdir(parents=True, exist_ok=True)
    with (dst / "pose_metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    meta = {"derived_from": src_name, "estimate": "lowest-loss stage-I seed (no distance alignment, beam or selection)", "n_views": len(out_rows)}
    if (src / "bench_meta.json").exists():
        meta.update({k: v for k, v in json.loads((src / "bench_meta.json").read_text(encoding="utf-8")).items() if k.startswith("VAL_")})
        meta["method"] = dst_name
    (dst / "bench_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ok = sum(float(r["rot_err_deg"]) <= 10 for r in out_rows)
    print(f"[ok] {dst}: {len(out_rows)} views, {ok} within 10 deg")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="")
    ap.add_argument("--src", default="ours_v10")
    ap.add_argument("--dst", default="ours_v10_stage1")
    ap.add_argument("--all", action="store_true", help="the five meshes of the paper")
    args = ap.parse_args()
    roots = [ROOT / "benchmark_runs" / d for d in ALL] if args.all else [Path(args.run_root)]
    return max(derive(r, args.src, args.dst) for r in roots)


if __name__ == "__main__":
    raise SystemExit(main())
