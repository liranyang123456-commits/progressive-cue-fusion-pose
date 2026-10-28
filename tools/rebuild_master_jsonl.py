#!/usr/bin/env python3
"""Rebuild benchmark_results/master.jsonl from all pose_metrics_summary.csv (6 metrics)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from record_benchmark_results import (  # noqa: E402
    MASTER_CSV,
    MASTER_JSONL,
    METHOD_META,
    METRIC_KEYS,
    RESULTS_DIR,
    _parse_run_name,
    collect_from_run_root,
    enrich_row,
)

PROD = ROOT / "accuracy_ref_view_prior_v9_run"


def main() -> int:
    rows: list[dict] = []
    br = ROOT / "benchmark_runs"
    if br.exists():
        for run_root in sorted(br.glob("*gt")):
            rows.extend(collect_from_run_root(run_root))

    # Production bunny lock (if not duplicate)
    prod_csv = PROD / "pose_metrics_summary.csv"
    if prod_csv.exists():
        import csv

        dataset, num_gt = "bunny", 10
        with prod_csv.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                tag = r.get("tag", "")
                if not tag:
                    continue
                er = enrich_row(r)
                key = f"{dataset}_{num_gt}gt_ours_v9_production_{tag}"
                mm = METHOD_META.get("ours_v9", {})
                row_out = {
                    "key": key,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "dataset": dataset,
                    "num_gt": num_gt,
                    "method": "ours_v9_production",
                    "tag": tag,
                    "family": "ours",
                    "paper": "VRHI v9 production lock",
                    "year": "2025",
                    "dr_backend": "nvdiffrast",
                    "output_dir": str(PROD),
                    "status": "ok",
                }
                for k in METRIC_KEYS:
                    row_out[k] = float(er[k])
                rows.append(row_out)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    backup = MASTER_JSONL.with_suffix(".jsonl.bak")
    if MASTER_JSONL.exists():
        MASTER_JSONL.replace(backup)
        print(f"[Backup] {backup}")

    with MASTER_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if rows:
        fields = list(rows[0].keys())
        import csv

        with MASTER_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    print(f"[Rebuild] {len(rows)} rows -> {MASTER_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
