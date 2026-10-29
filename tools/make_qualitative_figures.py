#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact qualitative figures for the revised manuscript (built from the strict no-GT runs only).

  fig_results_dense_gallery.png : five meshes x four views sampled at quantiles of the proposed method's e_R
                                  (10 / 40 / 70 / 95 %), target render next to the estimate render.
  fig_results_dense_compare.png : hard views; target vs. proposed vs. three DR baselines, every tile annotated
                                  with e_R and the failure mode (wrong shell / flipped view / local minimum).

  python tools/make_qualitative_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "benchmark_runs"
OUT = ROOT / "manuscript" / "computational_visual_media" / "figures" / "exp"
MESHES = [("bunny", "Bunny", 100), ("dragon", "Dragon", 100), ("armadillo", "Armadillo", 50), ("happy", "Happy Buddha", 50), ("lucy", "Lucy", 50)]
BASELINES = [("grid_adam", "Grid+Adam"), ("rand_restart_adam", "RandRestart+Adam"), ("adam_dr", "Adam-DR")]
TILE = 130  # tile height in px
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_rows(ds: str, n: int, method: str) -> dict[str, dict]:
    p = RUNS / f"{ds}_{n}gt" / method / "pose_metrics_summary.csv"
    if not p.exists():
        return {}
    rows = {}
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["tag"]] = r
    return rows


def mesh_radius(ds: str, n: int) -> float:
    """Bounding radius printed by the global-search stage of the reference run."""
    import re
    log = RUNS / f"{ds}_{n}gt" / "ours_v10" / "run.log"
    try:
        m = re.search(r"\br=([0-9.]+)\s+radii_mults", log.read_text(encoding="utf-8", errors="ignore"))
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return float("nan")


def failure_mode(r: dict, r_mesh: float) -> str:
    eR = float(r["rot_err_deg"])
    dd = abs(float(r["dist_err"]))
    dgt = float(r["gt_dist"])
    vde = float(r.get("view_dir_err_deg", eR))
    if eR <= 10:
        return "ok"
    if dd > 0.25 * dgt:
        return "wrong shell"
    if vde > 90 or eR > 120:
        return "flipped"
    if eR > 30:
        return "wrong basin"
    return "local min."


def read_tile(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        img = np.full((TILE, TILE, 3), 200, np.uint8)
    # auto-crop around the object (non-background), then resize to square tile
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = int(np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])))
    m = np.abs(g.astype(int) - bg) > 12
    ys, xs = np.where(m)
    if len(xs) > 20:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        half = int(max(x1 - x0, y1 - y0) * 0.6) + 6
        H, W = g.shape
        x0, x1 = max(0, cx - half), min(W, cx + half)
        y0, y1 = max(0, cy - half), min(H, cy + half)
        img = img[y0:y1, x0:x1]
    h, w = img.shape[:2]
    s = TILE / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((TILE, TILE, 3), 255, np.uint8)
    y, x = (TILE - img.shape[0]) // 2, (TILE - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def label(tile: np.ndarray, text: str, color=(0, 0, 0)) -> np.ndarray:
    """One or two text lines ('|' separates lines) under the tile, scaled to fit the tile width."""
    lines = text.split("|")
    bar = np.full((15 * len(lines) + 3, tile.shape[1], 3), 255, np.uint8)
    for i, ln in enumerate(lines):
        scale = 0.42
        while scale > 0.25 and cv2.getTextSize(ln, FONT, scale, 1)[0][0] > tile.shape[1] - 4:
            scale -= 0.02
        cv2.putText(bar, ln, (2, 12 + 15 * i), FONT, scale, color, 1, cv2.LINE_AA)
    return np.vstack([tile, bar])


def header(text: str, width: int, h: int = 18) -> np.ndarray:
    bar = np.full((h, width, 3), 235, np.uint8)
    cv2.putText(bar, text, (4, 13), FONT, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def hstack(tiles: list[np.ndarray], gap: int = 4) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    out = []
    for i, t in enumerate(tiles):
        if t.shape[0] < h:
            t = np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 255, np.uint8)])
        out.append(t)
        if i < len(tiles) - 1:
            out.append(np.full((h, gap, 3), 255, np.uint8))
    return np.hstack(out)


def vstack(rows: list[np.ndarray], gap: int = 6) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    out = []
    for i, r in enumerate(rows):
        if r.shape[1] < w:
            r = np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)])
        out.append(r)
        if i < len(rows) - 1:
            out.append(np.full((gap, w, 3), 255, np.uint8))
    return np.vstack(out)


def gallery() -> bool:
    rows_img = []
    for ds, disp, n in MESHES:
        ours = load_rows(ds, n, "ours_v10")
        if len(ours) < 4:
            continue
        r_mesh = mesh_radius(ds, n)
        tags = sorted(ours, key=lambda t: float(ours[t]["rot_err_deg"]))
        picks = [tags[min(len(tags) - 1, int(q * (len(tags) - 1)))] for q in (0.10, 0.40, 0.70, 0.95)]
        tiles = []
        for t in picks:
            d = RUNS / f"{ds}_{n}gt" / "ours_v10" / t
            r = ours[t]
            eR = float(r["rot_err_deg"])
            ddr = abs(float(r["dist_err"])) / r_mesh if np.isfinite(r_mesh) and r_mesh > 0 else float("nan")
            pair = hstack([label(read_tile(d / "gt_rgb.png"), "target"), label(read_tile(d / "best_rgb.png"), f"eR={eR:.2f}|dd/r={ddr:.3f}")], gap=2)
            tiles.append(pair)
        row = hstack(tiles, gap=10)
        rows_img.append(vstack([header(f"{disp} (N={len(ours)}): views at the 10/40/70/95 % quantiles of e_R", row.shape[1]), row], gap=2))
    if not rows_img:
        return False
    img = vstack(rows_img, gap=8)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "fig_results_dense_gallery.png"), img)
    return True


def compare() -> bool:
    rows_img = []
    for ds, disp, n in MESHES[:2]:
        ours = load_rows(ds, n, "ours_v10")
        base = {m: load_rows(ds, n, m) for m, _ in BASELINES}
        if not ours or not all(base.values()):
            continue
        r_mesh = mesh_radius(ds, n)
        common = [t for t in ours if all(t in base[m] for m, _ in BASELINES)]
        # views where at least one baseline fails (>10 deg); order by the number of failing baselines, then by ours' e_R
        def nfail(t):
            return sum(float(base[m][t]["rot_err_deg"]) > 10 for m, _ in BASELINES)
        hard = sorted([t for t in common if nfail(t) > 0], key=lambda t: (-nfail(t), float(ours[t]["rot_err_deg"])))
        ours_fail = sorted([t for t in common if float(ours[t]["rot_err_deg"]) > 10], key=lambda t: -float(ours[t]["rot_err_deg"]))
        picks = hard[:3] + [t for t in ours_fail if t not in hard[:3]][:1]
        if not picks:
            continue
        for t in picks:
            d0 = RUNS / f"{ds}_{n}gt" / "ours_v10" / t
            tiles = [label(read_tile(d0 / "gt_rgb.png"), f"{disp} {t}|target")]
            r = ours[t]
            fm = failure_mode(r, r_mesh)
            col = (0, 128, 0) if fm == "ok" else (0, 0, 200)
            tiles.append(label(read_tile(d0 / "best_rgb.png"), f"Proposed|eR={float(r['rot_err_deg']):.1f}  {fm}", col))
            for m, mname in BASELINES:
                rb = base[m][t]
                fmb = failure_mode(rb, r_mesh)
                colb = (0, 128, 0) if fmb == "ok" else (0, 0, 200)
                tiles.append(label(read_tile(RUNS / f"{ds}_{n}gt" / m / t / "best_rgb.png"), f"{mname}|eR={float(rb['rot_err_deg']):.1f}  {fmb}", colb))
            rows_img.append(hstack(tiles, gap=4))
    if not rows_img:
        return False
    img = vstack(rows_img, gap=6)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "fig_results_dense_compare.png"), img)
    return True


def main() -> int:
    g = gallery()
    c = compare()
    print(f"[fig] gallery={'ok' if g else 'no data'} compare={'ok' if c else 'no data'} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
