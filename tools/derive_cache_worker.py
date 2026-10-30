#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runs on the GPU worker: build <ds>_20gt/ours_v10 observation caches from the complete <ds>_100gt/ours_v10
caches (GT poses are a seeded sequence, so the first 20 views of the 100-view run are the 20-view run) and
complete the per-view cache files of a partially finished run copied from the control machine."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

CACHE_FILES = ("gt_refs_full.npz", "gt_refs_search.npz", "gt_rgb_ref.npy", "gt_camera_spherical.txt")


def tag(i: int) -> str:
    return f"gt_{i:02d}"


def complete(d: Path, n: int) -> bool:
    if not (d / "gt_poses.csv").exists():
        return False
    return all((d / tag(i) / "gt_refs_search.npz").exists() and (d / tag(i) / "gt_refs_full.npz").exists() for i in range(n))


def derive(runs: Path, ds: str, n: int, n_src: int) -> bool:
    src = runs / f"{ds}_{n_src}gt" / "ours_v10"
    dst = runs / f"{ds}_{n}gt" / "ours_v10"
    if complete(dst, n):
        print(f"[cache] {ds} N={n} already complete")
        return True
    if not complete(src, n_src):
        print(f"[cache] source {src} incomplete")
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (dst / tag(i)).mkdir(exist_ok=True)
        for f in CACHE_FILES:
            if (src / tag(i) / f).exists():
                shutil.copy2(src / tag(i) / f, dst / tag(i) / f)
    lines = (src / "gt_poses.csv").read_text(encoding="utf-8").splitlines()
    (dst / "gt_poses.csv").write_text("\n".join(lines[: n + 1]) + "\n", encoding="utf-8")
    print(f"[cache] derived {ds} N={n} from N={n_src}: {complete(dst, n)}")
    return complete(dst, n)


def fill_partial(runs: Path, ds: str, n: int, method: str) -> None:
    cache = runs / f"{ds}_{n}gt" / "ours_v10"
    out = runs / f"{ds}_{n}gt" / method
    if not out.exists():
        return
    k = 0
    for i in range(n):
        d = out / tag(i)
        if not d.exists():
            continue
        for f in CACHE_FILES:
            if not (d / f).exists() and (cache / tag(i) / f).exists():
                shutil.copy2(cache / tag(i) / f, d / f)
                k += 1
    for f in ("gt_poses.csv",):
        if not (out / f).exists() and (cache / f).exists():
            shutil.copy2(cache / f, out / f)
    rows = len([ln for ln in (out / "pose_metrics_summary.csv").read_text(encoding="utf-8").splitlines() if ln.strip()]) - 1 if (out / "pose_metrics_summary.csv").exists() else 0
    print(f"[partial] {ds}/{method}: {rows} finished rows, {k} cache files filled in, cache complete={complete(out, n)}")


def main() -> int:
    runs = Path(sys.argv[1]) / "benchmark_runs"
    ok = derive(runs, "bunny", 20, 100) and derive(runs, "dragon", 20, 100)
    fill_partial(runs, "bunny", 20, "ours_v10_1shell")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
