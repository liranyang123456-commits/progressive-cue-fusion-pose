#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download additional Stanford 3D Scanning Repository meshes and pre-simplify them
for the wide-baseline pose benchmark.

Targets (all public, https://graphics.stanford.edu/data/3Dscanrep/):
  armadillo   : Armadillo.ply.gz                      (~345k tris)
  happy       : happy_recon.tar.gz -> happy_vrip.ply  (~1.1M tris)
  lucy        : lucy.tar.gz -> lucy.ply               (~28M tris, 320 MB download)
  dragon_vrip : dragon_recon.tar.gz -> dragon_vrip.ply (~870k tris; the classic Stanford dragon)

When the Stanford server is unreachable (TLS handshake failures were observed from
this network) the script falls back to the public decimated mirror in
A. Jacobson's common-3d-test-models repository (same Stanford meshes, ~100k faces).

Each mesh is decimated to <= --target-tris triangles (quadric decimation, open3d)
and written to benchmark_assets/<name>_<k>k.ply. A JSON side-car records the
geometry descriptors used in the paper (bounding radius, triangle count, Euler
characteristic / genus proxy, extent ratios) so the dataset-selection rationale is
reproducible.

Usage:
  python tools/download_stanford_meshes.py --names armadillo,happy,lucy
  python tools/download_stanford_meshes.py --names lucy --target-tris 80000
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\reloc3r\3D_Standford_data")
ASSETS = ROOT / "benchmark_assets"

BASE = "https://graphics.stanford.edu/pub/3Dscanrep/"
MIRROR = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/"
SOURCES: dict[str, dict] = {
    "armadillo": {
        "url": BASE + "armadillo/Armadillo.ply.gz", "kind": "gz", "member": None, "out": "Armadillo.ply",
        "mirror": MIRROR + "armadillo.obj", "mirror_out": "armadillo_mirror.obj",
    },
    "happy": {
        "url": BASE + "happy/happy_recon.tar.gz", "kind": "tar", "member": "happy_recon/happy_vrip.ply", "out": "happy_vrip.ply",
        "mirror": MIRROR + "happy.obj", "mirror_out": "happy_mirror.obj",
    },
    "dragon_vrip": {
        "url": BASE + "dragon/dragon_recon.tar.gz", "kind": "tar", "member": "dragon_recon/dragon_vrip.ply", "out": "dragon_vrip.ply",
        "mirror": None, "mirror_out": None,
    },
    "lucy": {
        "url": BASE + "lucy.tar.gz", "kind": "tar", "member": "lucy.ply", "out": "lucy.ply",
        "mirror": MIRROR + "lucy.obj", "mirror_out": "lucy_mirror.obj",
    },
}


def _progress(name: str):
    t0 = time.time()

    def hook(count, block, total):
        done = count * block
        if total > 0 and (count % 200 == 0 or done >= total):
            pct = min(100.0, 100.0 * done / total)
            print(f"  [{name}] {pct:5.1f}%  {done/1e6:8.1f}/{total/1e6:.1f} MB  {time.time()-t0:6.0f}s", flush=True)

    return hook


def _fetch(url: str, dst: Path, name: str) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=_progress(name))
    tmp.replace(dst)


def download(name: str, force: bool = False) -> tuple[Path, str]:
    """Return (mesh_path, origin) where origin is 'stanford' or 'mirror'."""
    src = SOURCES[name]
    dst_dir = DATA_ROOT / name
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_ply = dst_dir / src["out"]
    out_mirror = dst_dir / src["mirror_out"] if src.get("mirror_out") else None
    if out_ply.exists() and not force:
        print(f"[skip] {out_ply} exists")
        return out_ply, "stanford"
    if out_mirror is not None and out_mirror.exists() and not force:
        print(f"[skip] {out_mirror} exists (mirror)")
        return out_mirror, "mirror"

    archive = dst_dir / Path(src["url"]).name
    try:
        if not archive.exists() or force:
            print(f"[download] {src['url']} -> {archive}")
            _fetch(src["url"], archive, name)
    except Exception as e:
        print(f"[warn] primary source failed: {e}")
        if not src.get("mirror") or out_mirror is None:
            raise
        print(f"[download] {src['mirror']} -> {out_mirror}")
        _fetch(src["mirror"], out_mirror, name)
        print(f"[ok] {out_mirror} ({out_mirror.stat().st_size/1e6:.1f} MB, mirror)")
        return out_mirror, "mirror"

    if src["kind"] == "gz":
        with gzip.open(archive, "rb") as f_in, out_ply.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            member = src["member"]
            names = tf.getnames()
            if member not in names:
                cand = [n for n in names if n.endswith(Path(member).name)]
                if not cand:
                    raise FileNotFoundError(f"{member} not in archive; members={names[:20]}")
                member = cand[0]
            with tf.extractfile(member) as f_in, out_ply.open("wb") as f_out:  # type: ignore[arg-type]
                shutil.copyfileobj(f_in, f_out)
    print(f"[ok] {out_ply} ({out_ply.stat().st_size/1e6:.1f} MB)")
    return out_ply, "stanford"


def describe_and_simplify(name: str, mesh_path: Path, origin: str, target_tris: int) -> dict:
    import numpy as np
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(mesh_path))
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    m.remove_unreferenced_vertices()
    tri = np.asarray(m.triangles)
    v = np.asarray(m.vertices)
    n_tri_full = int(tri.shape[0])
    center = v.mean(axis=0)
    radius = float(np.max(np.linalg.norm(v - center, axis=1)))
    ext = v.max(axis=0) - v.min(axis=0)
    edges = np.sort(tri[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    E = int(np.unique(edges, axis=0).shape[0])
    chi = int(v.shape[0]) - E + n_tri_full

    if n_tri_full > target_tris:
        ms = m.simplify_quadric_decimation(target_number_of_triangles=int(target_tris))
    else:
        ms = m
    ms.remove_degenerate_triangles()
    ms.remove_unreferenced_vertices()
    ms.compute_vertex_normals()
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"{name}_{target_tris//1000}k.ply"
    o3d.io.write_triangle_mesh(str(out), ms, write_ascii=False)
    desc = {
        "name": name,
        "source": SOURCES[name]["url"] if origin == "stanford" else SOURCES[name]["mirror"],
        "origin": origin,
        "input_mesh": str(mesh_path),
        "simplified_ply": str(out),
        "triangles_input": n_tri_full,
        "triangles_simplified": int(np.asarray(ms.triangles).shape[0]),
        "vertices_input": int(v.shape[0]),
        "bounding_radius": radius,
        "extent_xyz": [float(x) for x in ext],
        "aspect_max_over_min": float(ext.max() / max(ext.min(), 1e-9)),
        "euler_characteristic": chi,
        "genus_proxy": int(max(0, round((2 - chi) / 2))),
        "watertight": bool(ms.is_watertight()),
    }
    (ASSETS / f"{name}_{target_tris//1000}k.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")
    print(f"[simplified] {out}  tris {n_tri_full} -> {desc['triangles_simplified']}  r={radius:.4f}  chi={chi}")
    return desc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="armadillo,happy,lucy")
    ap.add_argument("--target-tris", type=int, default=80000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    report = []
    for n in names:
        if n not in SOURCES:
            print(f"[skip] unknown {n}")
            continue
        try:
            mesh_path, origin = download(n, force=args.force)
            report.append(describe_and_simplify(n, mesh_path, origin, args.target_tris))
        except Exception as e:  # keep going for the other meshes
            print(f"[error] {n}: {e}")
    out = ASSETS / "stanford_mesh_descriptors.json"
    existing = []
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    merged = {d["name"]: d for d in existing}
    for d in report:
        merged[d["name"]] = d
    out.write_text(json.dumps(list(merged.values()), indent=2), encoding="utf-8")
    print(f"[report] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
