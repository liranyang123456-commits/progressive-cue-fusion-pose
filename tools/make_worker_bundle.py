#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pack everything a GPU worker needs to run the synthetic benchmark on identical observations:
code, portable meshes and the ground-truth observation caches rendered on the control machine.

  python tools/make_worker_bundle.py --out G:/worker_bundle --runs bunny_100gt,dragon_100gt,armadillo_50gt,happy_50gt,lucy_50gt
"""
from __future__ import annotations
import os

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ["Exp_Camera_Optim_RT_Val_Beam_Search12291_Standford122.py", "run_benchmark_compare.py"]
CACHE_FILES = ("gt_refs_full.npz", "gt_refs_search.npz", "gt_rgb_ref.npy", "gt_camera_spherical.txt")
PORTABLE_SRC = {
    "bunny": Path(os.path.join(os.environ.get("STANFORD_ROOT", "data/stanford"), "bun_zipper.ply")),
    "dragon": ROOT / "benchmark_runs" / "dragon_100gt" / "ours_v10" / "mesh_exports" / "mesh_raw_o3d.ply",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default="bunny_100gt,dragon_100gt,armadillo_50gt,happy_50gt,lucy_50gt")
    ap.add_argument("--zip", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in CODE:
        shutil.copy2(ROOT / c, out / c)
    assets = out / "benchmark_assets"
    assets.mkdir(exist_ok=True)
    for name in ("armadillo_80k.ply", "happy_80k.ply", "lucy_80k.ply", "stanford_mesh_descriptors.json"):
        src = ROOT / "benchmark_assets" / name
        if src.exists():
            shutil.copy2(src, assets / name)
    for ds, src in PORTABLE_SRC.items():
        if src.exists():
            shutil.copy2(src, assets / f"{ds}_portable.ply")
            print(f"[bundle] {ds}_portable.ply <- {src} ({src.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"[bundle] WARNING missing portable source for {ds}: {src}")
    n_files = 0
    for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
        src_root = ROOT / "benchmark_runs" / run / "ours_v10"
        dst_root = out / "benchmark_runs" / run / "ours_v10"
        dst_root.mkdir(parents=True, exist_ok=True)
        for f in ("gt_poses.csv", "gt_poses_pose4x4.npy"):
            if (src_root / f).exists():
                shutil.copy2(src_root / f, dst_root / f)
        for tag_dir in sorted(src_root.glob("gt_*")):
            if not tag_dir.is_dir():
                continue
            (dst_root / tag_dir.name).mkdir(exist_ok=True)
            for f in CACHE_FILES:
                if (tag_dir / f).exists():
                    shutil.copy2(tag_dir / f, dst_root / tag_dir.name / f)
                    n_files += 1
        print(f"[bundle] {run}: {len(list(dst_root.glob('gt_*')))} views")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"[bundle] {n_files} cache files, total {total/1e9:.2f} GB -> {out}")
    if args.zip:
        zp = Path(args.zip)
        with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in out.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(out).as_posix())
        print(f"[bundle] zip {zp} ({zp.stat().st_size/1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
