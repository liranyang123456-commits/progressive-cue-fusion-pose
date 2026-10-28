#!/usr/bin/env python3
"""Aggregate pose_metrics_summary.csv rows into summary statistics."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_geom import full_pose_metrics_row  # noqa: E402

METRIC_KEYS = [
    "rot_err_deg",
    "cam_center_err",
    "view_dir_err_deg",
    "dist_err",
    "elev_err_deg",
    "azim_err_deg",
]

SR_THRESHOLDS = (5.0, 10.0, 15.0)

# Optional per-view stage timing columns (seconds), present in runs produced after
# the timing instrumentation was added to the main experiment script.
TIMING_KEYS = ["t_total_s", "t_global_s", "t_dist_align_s", "t_local_s", "t_select_s", "t_refine_s"]


def _timing_value(r: dict, k: str) -> float | None:
    v = r.get(k, "")
    if v is None or str(v).strip() in ("", "nan", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    pos = (len(s) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize_timing(rows: list[dict]) -> dict:
    """mean / median / P95 of each timing column over rows that carry it."""
    out: dict = {}
    for k in TIMING_KEYS:
        vals = [v for v in (_timing_value(r, k) for r in rows) if v is not None]
        if not vals:
            continue
        out[f"mean_{k}"] = sum(vals) / len(vals)
        out[f"median_{k}"] = _percentile(vals, 0.5)
        out[f"p95_{k}"] = _percentile(vals, 0.95)
        out[f"n_{k}"] = len(vals)
    return out


def _row_for_metrics(r: dict) -> dict:
    base = {
        "gt_dist": float(r["gt_dist"]),
        "gt_elev_deg": float(r["gt_elev_deg"]),
        "gt_azim_deg": float(r["gt_azim_deg"]),
    }
    if str(r.get("est_cam_x", "")).strip() not in ("", "nan"):
        base["est_cam_x"] = float(r["est_cam_x"])
        base["est_cam_y"] = float(r["est_cam_y"])
        base["est_cam_z"] = float(r["est_cam_z"])
    else:
        base["est_dist"] = float(r["est_dist"])
        base["est_elev_deg"] = float(r["est_elev_deg"])
        base["est_azim_deg"] = float(r["est_azim_deg"])
    return base


def enrich_row(r: dict) -> dict:
    try:
        float(r.get("cam_center_err", "nan"))
        return r
    except (TypeError, ValueError):
        pass
    m = full_pose_metrics_row(_row_for_metrics(r))
    out = dict(r)
    out.update(m)
    return out


def summarize(rows: list[dict]) -> dict:
    enriched = [enrich_row(r) for r in rows]
    n = len(enriched)
    out: dict = {"n": n}
    for k in METRIC_KEYS:
        vals = [float(r[k]) for r in enriched]
        out[f"mean_{k}"] = sum(vals) / n
        out[f"std_{k}"] = float(math.sqrt(sum((v - out[f"mean_{k}"]) ** 2 for v in vals) / max(n - 1, 1)))
        out[f"median_{k}"] = sorted(vals)[n // 2]
    for thr in SR_THRESHOLDS:
        ok = sum(1 for r in enriched if float(r["rot_err_deg"]) <= thr)
        out[f"sr_{int(thr)}deg"] = ok / n
    out.update(summarize_timing(rows))
    return out
