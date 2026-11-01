"""Shared helpers for running external pose estimators on the synthetic Stanford benchmark (tools/export_synth_frames.py).

Poses returned by the external methods are OpenCV object-to-camera poses of the re-centred mesh scaled to a
100 mm bounding radius; they are mapped back to the benchmark's native camera and scored with the same
quantities as pose_metrics_summary.csv of every other method (rotation error, camera-centre error, distance
error, viewing direction).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "benchmark_runs"
N_VIEWS = {"bunny": 100, "dragon": 100, "armadillo": 50, "happy": 50, "lucy": 50}
COLUMNS = ["tag", "rot_err_deg", "cam_center_err", "view_dir_err_deg", "gt_dist", "gt_elev_deg", "gt_azim_deg", "est_dist",
           "est_elev_deg", "est_azim_deg", "dist_err", "elev_err_deg", "azim_err_deg", "t_total_s", "t_global_s",
           "t_dist_align_s", "t_local_s", "t_select_s", "t_refine_s"]


def rot_err(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _sph(v: np.ndarray) -> tuple[float, float]:
    """elevation / azimuth of a direction in the benchmark's convention (y up, azimuth from +z towards +x)."""
    v = v / max(1e-12, float(np.linalg.norm(v)))
    return math.degrees(math.asin(float(np.clip(v[1], -1.0, 1.0)))), math.degrees(math.atan2(float(v[0]), float(v[2])))


def synth_metrics(R_est, t_est_mm, proto: dict) -> dict:
    R_est, t_est = np.asarray(R_est, dtype=np.float64), np.asarray(t_est_mm, dtype=np.float64)
    R_gt, t_gt = np.asarray(proto["gt_R"], dtype=np.float64), np.asarray(proto["gt_t"], dtype=np.float64)
    s = float(proto["scale_mm_per_unit"])
    c_est = -R_est.T @ t_est / s  # camera centre relative to the look-at centre, native units
    c_gt = -R_gt.T @ t_gt / s
    d_est, d_gt = float(np.linalg.norm(c_est)), float(np.linalg.norm(c_gt))
    cosv = float(np.dot(c_est, c_gt) / max(1e-12, d_est * d_gt))
    e_est, a_est = _sph(c_est)
    e_gt, a_gt = _sph(c_gt)
    da = abs((a_est - a_gt + 180.0) % 360.0 - 180.0)
    return {"rot_err_deg": rot_err(R_est, R_gt), "cam_center_err": float(np.linalg.norm(c_est - c_gt)),
            "view_dir_err_deg": math.degrees(math.acos(float(np.clip(cosv, -1.0, 1.0)))),
            "gt_dist": d_gt, "gt_elev_deg": e_gt, "gt_azim_deg": a_gt, "est_dist": d_est, "est_elev_deg": e_est,
            "est_azim_deg": a_est, "dist_err": abs(d_est - d_gt), "elev_err_deg": abs(e_est - e_gt), "azim_err_deg": da}


def failed_metrics(proto: dict) -> dict:
    """A frame on which the external method raised: counted as a failure (180 deg, distance error = distance)."""
    m = synth_metrics(np.asarray(proto["gt_R"]) @ np.diag([1.0, -1.0, -1.0]), proto["gt_t"], proto)
    m.update({"rot_err_deg": 180.0, "cam_center_err": 2.0 * m["gt_dist"], "est_dist": float("nan"), "dist_err": m["gt_dist"]})
    return m


class SummaryWriter:
    """Appends rows to benchmark_runs/<mesh>_<N>gt/<method>/pose_metrics_summary.csv (resumable by tag)."""

    def __init__(self, ds: str, method: str, machine: str = "RTX 5090") -> None:
        self.dir = RUNS / f"{ds}_{N_VIEWS[ds]}gt" / method
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv = self.dir / "pose_metrics_summary.csv"
        (self.dir / "machine.txt").write_text(machine, encoding="utf-8")
        self.done = set()
        if self.csv.exists():
            self.done = {r["tag"] for r in csv.DictReader(self.csv.open(encoding="utf-8"))}
        else:
            with self.csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(COLUMNS)

    def write(self, tag: str, met: dict, t_total: float, extra: dict | None = None) -> None:
        row = {**met, "tag": tag, "t_total_s": t_total, "t_global_s": "", "t_dist_align_s": "", "t_local_s": "",
               "t_select_s": "", "t_refine_s": t_total}
        with self.csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(c, "") for c in COLUMNS])
        with (self.dir / "records.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"tag": tag, **met, "t_total_s": t_total, **(extra or {})}) + "\n")
        self.done.add(tag)