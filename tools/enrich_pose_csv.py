#!/usr/bin/env python3
"""Backfill pose_metrics_summary.csv with full 6-metric columns (+ optional P3D rows)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_geom import full_pose_metrics_row  # noqa: E402

STANDARD_FIELDS = [
    "tag",
    "rot_err_deg",
    "cam_center_err",
    "view_dir_err_deg",
    "gt_dist",
    "gt_elev_deg",
    "gt_azim_deg",
    "est_dist",
    "est_elev_deg",
    "est_azim_deg",
    "dist_err",
    "elev_err_deg",
    "azim_err_deg",
    # per-view stage timing (seconds), written by the main experiment script
    "t_total_s",
    "t_global_s",
    "t_dist_align_s",
    "t_local_s",
    "t_select_s",
    "t_refine_s",
]


def _needs_enrich(row: dict) -> bool:
    try:
        float(row.get("cam_center_err", "nan"))
        return False
    except (TypeError, ValueError):
        return True


def enrich_row(row: dict) -> dict:
    base = {
        "tag": row.get("tag", ""),
        "gt_dist": float(row["gt_dist"]),
        "gt_elev_deg": float(row["gt_elev_deg"]),
        "gt_azim_deg": float(row["gt_azim_deg"]),
    }
    if str(row.get("est_cam_x", "")).strip() not in ("", "nan"):
        base["est_cam_x"] = float(row["est_cam_x"])
        base["est_cam_y"] = float(row["est_cam_y"])
        base["est_cam_z"] = float(row["est_cam_z"])
    else:
        base["est_dist"] = float(row["est_dist"])
        base["est_elev_deg"] = float(row["est_elev_deg"])
        base["est_azim_deg"] = float(row["est_azim_deg"])

    m = full_pose_metrics_row(base)
    out = {**row, **base, **m}
    for k in ("est_dist", "est_elev_deg", "est_azim_deg"):
        if k not in out and k in base:
            out[k] = base[k]
    return out


def process_csv(path: Path, dry_run: bool = False) -> bool:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        return False
    if not any(_needs_enrich(r) for r in rows) and "cam_center_err" in rows[0]:
        return False
    enriched = [enrich_row(r) for r in rows]
    if dry_run:
        print(f"[dry] {path}")
        return True
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STANDARD_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(enriched)
    print(f"[ok] {path}")
    return True


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(ROOT / "benchmark_runs" / "bunny_10gt"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.run_root)
    n = 0
    for csv_path in sorted(root.glob("*/pose_metrics_summary.csv")):
        if process_csv(csv_path, args.dry_run):
            n += 1
    prod = ROOT / "accuracy_ref_view_prior_v9_run" / "pose_metrics_summary.csv"
    if prod.exists() and process_csv(prod, args.dry_run):
        n += 1
    print(f"Enriched {n} files under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
