#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot assembly of the revised submission from the raw run outputs.

  python tools/finalize_revision.py [--skip-geom] [--no-compile]

Steps: derive Geom-only rows -> merge worker results (if present) -> tables/macros/figures -> BOP tables ->
leak-test table -> compile main_pr.tex, supplementary_pr.tex, response_to_reviewers.tex -> report page counts
and the list of %% VERIFY markers still present in the manuscript sources.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manuscript" / "computational_visual_media"
PY = sys.executable


def run(cmd: list[str], cwd: Path = ROOT) -> int:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def merge_worker_results(worker_root: Path) -> None:
    """Copy pose_metrics_summary.csv (+ per-view dirs) produced on the GPU worker into benchmark_runs/.
    Only runs that do not exist locally (or have fewer rows) are taken."""
    if not worker_root.exists():
        return
    for csv in worker_root.glob("benchmark_runs/*/*/pose_metrics_summary.csv"):
        rel = csv.relative_to(worker_root)
        dst = ROOT / rel
        n_src = len(csv.read_text(encoding="utf-8").splitlines())
        n_dst = len(dst.read_text(encoding="utf-8").splitlines()) if dst.exists() else 0
        if n_src > n_dst:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv, dst)
            (dst.parent / "machine.txt").write_text("RTX 2080 Ti", encoding="utf-8")
            for extra in ("bench_meta.json", "gt_poses.csv", "run.log"):
                if (csv.parent / extra).exists():
                    shutil.copy2(csv.parent / extra, dst.parent / extra)
            for tag_dir in csv.parent.glob("gt_*"):
                if tag_dir.is_dir():
                    (dst.parent / tag_dir.name).mkdir(exist_ok=True)
                    for f in tag_dir.iterdir():
                        if f.suffix in (".json", ".txt", ".png", ".npy", ".csv") and not f.name.startswith("gt_refs"):
                            shutil.copy2(f, dst.parent / tag_dir.name / f.name)
                    for f in tag_dir.glob("seed_00/*_rand_octant_best.json"):
                        (dst.parent / tag_dir.name / "seed_00").mkdir(exist_ok=True)
                        shutil.copy2(f, dst.parent / tag_dir.name / "seed_00" / f.name)
            print(f"[merge] {rel} ({n_src-1} rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-root", default=str(ROOT / "worker_results"), help="synced copy of the worker's pose_bench directory")
    ap.add_argument("--skip-geom", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    merge_worker_results(Path(args.worker_root))
    if not args.skip_geom:
        for run_root in sorted((ROOT / "benchmark_runs").glob("*_*gt")):
            if (run_root / "ours_v10" / "pose_metrics_summary.csv").exists():
                run([PY, str(ROOT / "tools" / "derive_geom_only.py"), "--run-root", str(run_root)])
    run([PY, str(ROOT / "tools" / "export_revision_results.py")])
    run([PY, str(ROOT / "tools" / "export_bop_results.py")])
    run([PY, str(ROOT / "tools" / "export_leak_table.py")])
    run([PY, str(ROOT / "tools" / "make_qualitative_figures.py")])
    if args.no_compile:
        return 0
    for doc in ("main_pr", "supplementary_pr", "response_to_reviewers"):
        if not (MAN / f"{doc}.tex").exists():
            continue
        for i in range(3):
            rc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{doc}.tex"], cwd=str(MAN), capture_output=True).returncode
            if i == 0 and doc == "main_pr":
                subprocess.run(["bibtex", doc], cwd=str(MAN), capture_output=True)
            if rc != 0:
                break
        log = (MAN / f"{doc}.log").read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"Output written on .*?\((\d+) pages", log)
        errs = [ln for ln in log.splitlines() if ln.startswith("!")]
        undef = sorted(set(re.findall(r"Reference `([^']+)' on page", log)))
        print(f"[compile] {doc}: {'OK' if rc == 0 else 'FAILED'}; pages={m.group(1) if m else '?'}; errors={errs[:2]}; undefined refs={undef}")
    verify = []
    for tex in MAN.glob("*.tex"):
        for i, ln in enumerate(tex.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if "VERIFY" in ln:
                verify.append(f"{tex.name}:{i}: {ln.strip()[:100]}")
    print(f"[verify-markers] {len(verify)} remaining")
    for v in verify:
        print("   ", v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
