#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan benchmark_runs/ and append to benchmark_results/master.jsonl + REPORT.md.

Usage:
  python record_benchmark_results.py --scan-all
  python record_benchmark_results.py --run-root benchmark_runs/bunny_10gt
  python record_benchmark_results.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tools.benchmark_aggregate import (  # noqa: E402
    METRIC_KEYS,
    SR_THRESHOLDS,
    TIMING_KEYS,
    _timing_value,
    enrich_row,
    summarize,
)
RESULTS_DIR = ROOT / "benchmark_results"
MASTER_JSONL = RESULTS_DIR / "master.jsonl"
MASTER_CSV = RESULTS_DIR / "master.csv"
REPORT_MD = RESULTS_DIR / "REPORT.md"

METHOD_META: dict[str, dict[str, str]] = {
    "ours_v9": {"family": "ours", "paper": "VRHI v9 RandMultiR+Beam", "year": "2025", "dr_backend": "nvdiffrast"},
    "ours_v9_se3": {"family": "ours", "paper": "v9 + SE3 refine", "year": "2025", "dr_backend": "nvdiffrast"},
    "adam_dr": {"family": "dr", "paper": "PyTorch3D/nvdiffrast pose Adam", "year": "2020", "dr_backend": "nvdiffrast_aa"},
    "adam_dr_hard": {"family": "dr", "paper": "Hard raster DR", "year": "2020", "dr_backend": "nvdiffrast_hard"},
    "adam_dr_gauss_soft": {"family": "dr_soft", "paper": "Gaussian-soft sil (SoftRas-like)", "year": "2019", "dr_backend": "gaussian_soft"},
    "adam_dr_sil_only": {"family": "dr", "paper": "Silhouette-only DR", "year": "2020", "dr_backend": "nvdiffrast_aa"},
    "hybrid_dr": {"family": "dr", "paper": "Greedy+Adam (nvdiffrast pose)", "year": "2020", "dr_backend": "nvdiffrast_aa"},
    "hybrid_dr_gauss_soft": {"family": "dr_soft", "paper": "Hybrid+Gaussian soft sil", "year": "2019", "dr_backend": "gaussian_soft"},
    "grid_adam": {"family": "dr_search", "paper": "Grid search + Adam DR", "year": "2020", "dr_backend": "nvdiffrast_aa"},
    "grid_adam_gauss_soft": {"family": "dr_soft", "paper": "Grid + Adam + soft sil", "year": "2019", "dr_backend": "gaussian_soft"},
    "grid_hybrid": {"family": "dr_search", "paper": "Grid + Hybrid DR", "year": "2020", "dr_backend": "nvdiffrast_aa"},
    "rand_restart_adam": {"family": "dr_search", "paper": "Random-restart Adam DR", "year": "2020", "dr_backend": "nvdiffrast_aa"},
    "greedy_bt_local": {"family": "ablation", "paper": "Global+greedy_bt", "year": "2025", "dr_backend": "nvdiffrast"},
    "rand_topk_only": {"family": "ablation", "paper": "Global TopK only", "year": "2025", "dr_backend": "nvdiffrast"},
    "ours_v9_geom": {"family": "ours", "paper": "v9 geom-only select (no VisualSelect)", "year": "2025", "dr_backend": "nvdiffrast"},
    "ours_v12c": {"family": "ours", "paper": "v12c Beam+cond END/DR elev>=58", "year": "2025", "dr_backend": "nvdiffrast"},
    "ours_v12f": {"family": "ours", "paper": "v12f GT-elev cond END/DR", "year": "2025", "dr_backend": "nvdiffrast"},
    "grid_adam_top16": {"family": "dr_search", "paper": "Grid+Adam Top16 MS600", "year": "2020", "dr_backend": "nvdiffrast_aa"},
    "ours_v10": {"family": "ours", "paper": "Progressive cue fusion (full)", "year": "2026", "dr_backend": "nvdiffrast"},
    "ours_v10_noda": {"family": "ablation", "paper": "v10 w/o distance alignment", "year": "2026", "dr_backend": "nvdiffrast"},
    "ours_v10_geom": {"family": "ablation", "paper": "v10 geometry-only selection", "year": "2026", "dr_backend": "nvdiffrast"},
    "ours_v10_fixedmr": {"family": "ablation", "paper": "v10 fixed multi-radius (36 shells, equal budget)", "year": "2026", "dr_backend": "nvdiffrast"},
    "ours_v10_1shell": {"family": "ablation", "paper": "v10 single shell 2r (equal budget)", "year": "2026", "dr_backend": "nvdiffrast"},
    "pytorch3d_softsil": {"family": "dr_external", "paper": "PyTorch3D SoftSilhouette", "year": "2020", "dr_backend": "pytorch3d"},
    "p3d_softsil_spherical": {"family": "dr_p3d", "paper": "PyTorch3D tutorial SoftSil spherical", "year": "2020", "dr_backend": "pytorch3d"},
    "p3d_softsil_xyz": {"family": "dr_p3d", "paper": "PyTorch3D tutorial camera_position xyz", "year": "2020", "dr_backend": "pytorch3d"},
    "p3d_softsil_grid": {"family": "dr_p3d", "paper": "PyTorch3D SoftSil + grid init", "year": "2020", "dr_backend": "pytorch3d"},
    "p3d_softsil_reg": {"family": "dr_p3d", "paper": "PyTorch3D SoftSil + mesh reg", "year": "2020", "dr_backend": "pytorch3d"},
    "p3d_softsil_gt_init": {"family": "dr_oracle", "paper": "PyTorch3D GT init (upper bound)", "year": "2020", "dr_backend": "pytorch3d"},
    "softras_iccv19": {"family": "dr_sota", "paper": "Soft Rasterizer Liu ICCV19", "year": "2019", "dr_backend": "soft_renderer"},
    "softras_iccv19_grid": {"family": "dr_sota", "paper": "SoftRas + grid init (fair)", "year": "2019", "dr_backend": "soft_renderer"},
}


def _parse_run_name(name: str) -> tuple[str, int]:
    # bunny_10gt -> bunny, 10
    if "_" in name and name.endswith("gt"):
        base, ng = name.rsplit("_", 1)
        try:
            n = int(ng.replace("gt", ""))
            return base, n
        except ValueError:
            pass
    return name, 0


def _load_existing_keys() -> set[str]:
    keys: set[str] = set()
    if not MASTER_JSONL.exists():
        return keys
    with MASTER_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                keys.add(row.get("key", ""))
            except json.JSONDecodeError:
                continue
    return keys


def collect_from_run_root(run_root: Path) -> list[dict]:
    rows: list[dict] = []
    dataset, num_gt = _parse_run_name(run_root.name)
    for method_dir in sorted(run_root.iterdir()):
        if not method_dir.is_dir():
            continue
        csv_path = method_dir / "pose_metrics_summary.csv"
        if not csv_path.exists():
            continue
        meta_path = method_dir / "bench_meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        method = method_dir.name
        mm = METHOD_META.get(method, {"family": "unknown", "paper": method, "year": "", "dr_backend": ""})
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                tag = r.get("tag") or r.get("Tag") or ""
                if not tag:
                    continue
                key = f"{dataset}_{num_gt}gt_{method}_{tag}"
                er = enrich_row(r)
                row_out = {
                    "key": key,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "dataset": dataset,
                    "num_gt": num_gt,
                    "method": method,
                    "tag": tag,
                    "family": mm.get("family", ""),
                    "paper": mm.get("paper", ""),
                    "year": mm.get("year", ""),
                    "dr_backend": meta.get("VAL_SIL_RENDER_MODE", mm.get("dr_backend", "")),
                    "output_dir": str(method_dir),
                    "status": "ok",
                }
                for k in METRIC_KEYS:
                    row_out[k] = float(er[k])
                for k in TIMING_KEYS:
                    tv = _timing_value(r, k)
                    if tv is not None:
                        row_out[k] = tv
                rows.append(row_out)
    return rows


def append_rows(rows: list[dict]) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_keys()
    new_rows = [r for r in rows if r["key"] not in existing]
    if not new_rows:
        return 0
    with MASTER_JSONL.open("a", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new_rows)


def export_master_csv() -> None:
    if not MASTER_JSONL.exists():
        return
    all_rows: list[dict] = []
    with MASTER_JSONL.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_rows.append(json.loads(line))
    if not all_rows:
        return
    fields = list(all_rows[0].keys())
    with MASTER_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)


def _summaries_from_run(run_root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for method_dir in sorted(run_root.iterdir()):
        if not method_dir.is_dir():
            continue
        csv_path = method_dir / "pose_metrics_summary.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if rows:
            out[method_dir.name] = summarize(rows)
    prod = ROOT / "accuracy_ref_view_prior_v9_run" / "pose_metrics_summary.csv"
    if run_root.name == "bunny_10gt" and prod.exists():
        with prod.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if rows:
            out["ours_v9_production"] = summarize(rows)
    return out


def write_report(ref_method: str = "ours_v9") -> None:
    lines = [
        "# Benchmark 综合报告",
        "",
        f"生成时间 (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        f"主日志: `{MASTER_JSONL}`（含 6 项 pose 指标 / 视角）",
        "",
        "指标: $e_R$ (rot), $e_t$ (cam_center), $e_v$ (view_dir), $|\\Delta d|$, $|\\Delta$elev, $|\\Delta$azim; SR@5/10/15° = $e_R$ 成功率。",
        "",
    ]

    br = ROOT / "benchmark_runs"
    for run_root in sorted(br.glob("*gt")) if br.exists() else []:
        sums = _summaries_from_run(run_root)
        if not sums:
            continue
        ref = "ours_v9_production" if "ours_v9_production" in sums else ref_method
        ref_er = sums.get(ref, {}).get("mean_rot_err_deg")

        lines.append(f"## {run_root.name}")
        lines.append("")
        lines.append("### 主表：平均 $e_R$ (°)")
        lines.append("")
        lines.append("| 方法 | mean $e_R$ | std | SR@5° | SR@10° | SR@15° | family |")
        lines.append("|------|------------|-----|-------|--------|--------|--------|")
        for m, s in sorted(sums.items(), key=lambda x: x[1]["mean_rot_err_deg"]):
            mm = METHOD_META.get(m, {})
            lines.append(
                f"| {m} | {s['mean_rot_err_deg']:.3f} | {s['std_rot_err_deg']:.3f} | "
                f"{100*s['sr_5deg']:.0f}% | {100*s['sr_10deg']:.0f}% | {100*s['sr_15deg']:.0f}% | {mm.get('family','')} |"
            )
        lines.append("")

        lines.append("### 全指标均值（完整 $N$ 的方法）")
        lines.append("")
        hdr = "| 方法 | " + " | ".join(
            k.replace("_deg", "°").replace("rot_err", "e_R").replace("cam_center_err", "e_t")
            .replace("view_dir_err", "e_v").replace("dist_err", "|Δd|").replace("elev_err", "|Δelev|")
            .replace("azim_err", "|Δazim|")
            for k in METRIC_KEYS
        ) + " |"
        lines.append(hdr)
        lines.append("|" + "------|" * (len(METRIC_KEYS) + 1))
        for m, s in sorted(sums.items(), key=lambda x: x[1]["mean_rot_err_deg"]):
            if s["n"] < 10 and m not in ("ours_v9_production",):
                continue
            cells = " | ".join(f"{s[f'mean_{k}']:.3f}" for k in METRIC_KEYS)
            lines.append(f"| {m} | {cells} |")
        lines.append("")

        # Per-view runtime (seconds) when the run carries stage timing columns.
        timed = {m: s for m, s in sums.items() if "mean_t_total_s" in s}
        if timed:
            lines.append("### 运行时间（每视角，秒；同一 GPU）")
            lines.append("")
            lines.append("| 方法 | n | total mean | median | P95 | global | dist-align | local | select | refine |")
            lines.append("|------|---|------------|--------|-----|--------|------------|-------|--------|--------|")
            for m, s in sorted(timed.items(), key=lambda x: x[1]["mean_t_total_s"]):
                def _g(key: str) -> str:
                    v = s.get(key)
                    return f"{v:.2f}" if isinstance(v, (int, float)) else "-"
                lines.append(
                    f"| {m} | {s.get('n_t_total_s', s['n'])} | {_g('mean_t_total_s')} | {_g('median_t_total_s')} | "
                    f"{_g('p95_t_total_s')} | {_g('mean_t_global_s')} | {_g('mean_t_dist_align_s')} | "
                    f"{_g('mean_t_local_s')} | {_g('mean_t_select_s')} | {_g('mean_t_refine_s')} |"
                )
            lines.append("")

        lines.append("### SOTA / 外部 DR 子集（论文 Table SOTA）")
        lines.append("")
        sota_ids = {
            "ours_v9_production",
            "ours_v9",
            "grid_adam",
            "rand_restart_adam",
            "adam_dr",
            "adam_dr_gauss_soft",
            "p3d_softsil_grid",
            "p3d_softsil_spherical",
            "p3d_softsil_xyz",
            "softras_iccv19",
            "softras_iccv19_grid",
        }
        lines.append("| 方法 | 文献 | $e_R$↓ | $e_t$↓ | SR@10° |")
        lines.append("|------|------|--------|--------|--------|")
        for m in sorted(sota_ids):
            if m not in sums:
                continue
            s = sums[m]
            mm = METHOD_META.get(m, {"paper": m, "year": ""})
            delta = ""
            if ref_er is not None and m != ref:
                delta = f" ({s['mean_rot_err_deg'] - ref_er:+.1f})"
            lines.append(
                f"| {m} | {mm.get('paper','')} ({mm.get('year','')}) | "
                f"{s['mean_rot_err_deg']:.2f}{delta} | {s['mean_cam_center_err']:.3f} | {100*s['sr_10deg']:.0f}% |"
            )
        lines.append("")

    lines.append("## 文献索引")
    lines.append("")
    lines.append("详见 `DR_METHODS_LITERATURE.md`、`SOTA_COMPARISON_MATRIX.md`。")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Report] wrote {REPORT_MD}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-all", action="store_true", help="Scan benchmark_runs/")
    ap.add_argument("--run-root", default="", help="Single run folder e.g. benchmark_runs/bunny_10gt")
    ap.add_argument("--report", action="store_true", help="Regenerate REPORT.md from master.jsonl")
    ap.add_argument("--export-csv", action="store_true", help="Export master.csv")
    ap.add_argument("--rebuild-master", action="store_true", help="Rebuild master.jsonl from all CSVs (6 metrics)")
    args = ap.parse_args()

    if args.rebuild_master:
        subprocess.run([sys.executable, str(ROOT / "tools" / "rebuild_master_jsonl.py")], cwd=str(ROOT), check=False)

    if args.scan_all:
        br = ROOT / "benchmark_runs"
        total = 0
        if br.exists():
            for run_root in sorted(br.iterdir()):
                if run_root.is_dir():
                    n = append_rows(collect_from_run_root(run_root))
                    total += n
                    print(f"[Record] {run_root.name}: +{n} rows")
        print(f"[Record] total new rows: {total}")
    elif args.run_root:
        n = append_rows(collect_from_run_root(Path(args.run_root)))
        print(f"[Record] +{n} rows from {args.run_root}")
    elif not (args.report or args.export_csv):
        ap.print_help()
        return 0

    if args.export_csv or args.scan_all or args.run_root:
        export_master_csv()
    if args.report or args.scan_all or args.run_root:
        write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
