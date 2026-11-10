#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproduce the scale-adaptive multi-radius sampling of the proposed method
(ours_v10) and render a 3D visualization that matches the real runtime
parameters verified from the benchmark logs:

  - M = 40 log-spaced radius shells over the default range [0.4, 6.0] r, which the
        depth hint only widens (Eq. (4c) of the paper): on the dragon the default
        range is used in 97 of the 100 views and the inner bound drops to 0.3 r in
        the three views with the smallest hint (bunny: 94 and 6 of 100)
  - S = 1024 directions per radius (8 octants x 128, Gaussian + sign mirror)
  - direction set generated ONCE on the first octant and reused on every shell
        ("scale amplification")
  - deterministic elevation/azimuth grid anchors on every shell
  - real ground-truth camera positions from gt_poses.csv overlaid, to show that
    the correct shell lies INSIDE the sampled range (the w/o-DA fixed range
    [0.5, 4.0] r is also drawn for contrast).

All numbers below come from:
  - Exp_Camera_Optim_..._Standford122.py : _resolve_radii_mults
        (VAL_RAND_MIN_RADIUS_MULT=0.4, VAL_RAND_MAX_RADIUS_MULT=6.0, widened by the
        hint only), _resolve_samples_per_radius, _sample_dirs_in_octant_uniform,
        VAL_RAND_ANCHOR_ELEVS/AZIMS, VAL_RAND_RADIUS_STEPS=40
  - benchmark_runs/dragon_100gt/ours_v10/run.log : radii_mults=[0.4,...,6.0] in 97
        views and [0.3,...,6.0] in 3, samples_per_radius=1024 (per_octant=128),
        r=119.39
  - benchmark_runs/bunny_100gt/ours_v10/run.log : radii_mults=[0.4,...,6.0] in 94
        views and [0.3,...,6.0] in 6
  - benchmark_runs/<mesh>_100gt/ours_v10/gt_poses.csv : real GT camera centers

  python plot_sampling_space.py [--gt-csv PATH] [--out PATH]
"""
import argparse
import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

RNG_SEED = 12291  # the fixed benchmark seed (verified from run.log)

# ---------------------------------------------------------------------------
# 1. Reproduce _resolve_radii_mults exactly (dragon v10 adaptive branch)
# ---------------------------------------------------------------------------
def resolve_radii_mults(min_m, max_m, n_steps=40):
    """Log-spaced multipliers, identical to _resolve_radii_mults L4202-4203."""
    log_min = math.log(max(1e-3, min_m))
    log_max = math.log(max(min_m + 1e-3, max_m))
    return [math.exp(log_min + i * (log_max - log_min) / max(1, n_steps - 1))
            for i in range(n_steps)]

# Default range [0.4, 6.0] of the proposed method, the one used in 97 of the 100
# dragon views (see the module docstring). We draw the dragon (the harder, more
# illustrative case) with r = 119.39 from the log.
R_MESH = 119.393555
RANGE_V10 = (0.4, 6.0)
MULTS_V10 = resolve_radii_mults(*RANGE_V10, n_steps=40)      # ours (default range)
MULTS_V9  = resolve_radii_mults(0.5, 4.0, n_steps=40)        # w/o-DA (fixed)
M = len(MULTS_V10)

# ---------------------------------------------------------------------------
# 2. Reproduce _sample_dirs_in_octant_uniform (Gaussian + sign mirror, 8 octants)
# ---------------------------------------------------------------------------
def sample_directions_8octants(n_per_octant, rng):
    """Mirror first-octant Gaussian directions into 8 octants (L4528-4536)."""
    octants = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
               (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]
    v = np.abs(rng.standard_normal((n_per_octant, 3)))
    v = v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-8)
    dirs = []
    for (sx, sy, sz) in octants:
        s = np.array([sx, sy, sz], dtype=float).reshape(1, 3)
        d = v * s
        d = d / np.linalg.norm(d, axis=1, keepdims=True).clip(min=1e-8)
        dirs.append(d)
    return np.concatenate(dirs, axis=0)   # (8 * n_per_octant, 3)

rng = np.random.default_rng(RNG_SEED)
N_PER_OCT = 128                                   # per_octant=128 from log
BASE_DIRS = sample_directions_8octants(N_PER_OCT, rng)   # (1024, 3), reused
S = BASE_DIRS.shape[0]
assert S == 1024, S

# deterministic anchors (VAL_RAND_ANCHOR_ELEVS / AZIMS grid), reused per shell
ANCHOR_ELEVS = [-60, -30, 0, 30, 60]
ANCHOR_AZIMS = [-150, -90, -30, 30, 90, 150]
anchor_dirs = []
for el in ANCHOR_ELEVS:
    for az in ANCHOR_AZIMS:
        elr, azr = math.radians(el), math.radians(az)
        d = np.array([math.cos(elr) * math.cos(azr),
                      math.cos(elr) * math.sin(azr),
                      math.sin(elr)])
        anchor_dirs.append(d / np.linalg.norm(d))
ANCHOR_DIRS = np.array(anchor_dirs)

# ---------------------------------------------------------------------------
# 3. Load real GT camera centers from gt_poses.csv (dragon)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
_GT_REL = os.path.join("benchmark_runs", "dragon_100gt", "ours_v10", "gt_poses.csv")
_GT_DEFAULT = next((p for p in (os.path.join(ROOT, _GT_REL), os.path.join(ROOT, "results", _GT_REL))
                    if os.path.exists(p)), os.path.join(ROOT, _GT_REL))
_OUT_DEFAULT = os.path.join(ROOT, "manuscript", "computational_visual_media", "figures", "fig_sampling_space.png")
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--gt-csv", default=_GT_DEFAULT, help="gt_poses.csv of the dragon benchmark run")
_ap.add_argument("--out", default=_OUT_DEFAULT, help="output PNG")
_args = _ap.parse_args()
GT_CSV = _args.gt_csv
gt_centers = []
with open(GT_CSV, newline="") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        # camera center o = -R^T t ; columns m03,m13,m23 are translation t in
        # world frame (look-at places camera at o = a + d*u; the stored 4x4 is
        # world->camera, so camera center in world = -(R^T) t + a). With a=0
        # this is simply the negative of the translation rotated back; but the
        # distance/elev/azim already parametrize o directly: o = d*u(elev,azim).
        d = float(row["dist"]); el = math.radians(float(row["elev_deg"]))
        az = math.radians(float(row["azim_deg"]))
        u = np.array([math.cos(el) * math.cos(az),
                      math.cos(el) * math.sin(az),
                      math.sin(el)])
        gt_centers.append(d * u)
GT_CENTERS = np.array(gt_centers)   # (100, 3)

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 9), dpi=180)
ax = fig.add_subplot(111, projection="3d")

# draw the proposed (v10) shells: a subset for clarity, colored by shell index
show_idx = np.linspace(0, M - 1, 8).astype(int)   # 8 representative shells
cmap = plt.cm.viridis
for k, i in enumerate(show_idx):
    m = MULTS_V10[i]
    radius = m * R_MESH
    # translucent sphere wireframe
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=cmap(k / (len(show_idx) - 1)),
                      alpha=0.12, linewidth=0.5, rstride=4, cstride=6)

# plot sampled directions on a few representative shells (reuse BASE_DIRS)
for k, i in enumerate(show_idx):
    m = MULTS_V10[i]
    radius = m * R_MESH
    pts = BASE_DIRS * radius
    col = cmap(k / (len(show_idx) - 1))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[col], s=2.2,
               alpha=0.55, depthshade=False, edgecolors="none")
    # anchors
    apts = ANCHOR_DIRS * radius
    ax.scatter(apts[:, 0], apts[:, 1], apts[:, 2], c=[col], s=14,
               marker="^", alpha=0.9, depthshade=False, edgecolors="black",
               linewidths=0.3)

# v9 (w/o-DA) fixed range boundary: inner [0.5]r and outer [4.0]r spheres
for m, ls in [(0.5, "--"), (4.0, "--")]:
    radius = m * R_MESH
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#d62728", alpha=0.22, linewidth=0.8,
                      rstride=6, cstride=8, linestyle=ls)

# ground-truth camera positions (real data)
ax.scatter(GT_CENTERS[:, 0], GT_CENTERS[:, 1], GT_CENTERS[:, 2],
           c="red", s=26, marker="*", alpha=1.0, depthshade=False,
           edgecolors="darkred", linewidths=0.4, zorder=10)

# mesh center
ax.scatter([0], [0], [0], c="black", s=40, marker="o", zorder=11)

ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Scale-adaptive multi-radius sampling on the viewing sphere\n"
             "(dragon, r=%.1f; M=%d shells, S=%d dirs/shell, 8 octants)"
             % (R_MESH, M, S), fontsize=12)

# equal aspect
allpts = np.concatenate([BASE_DIRS * MULTS_V10[-1] * R_MESH,
                         GT_CENTERS], axis=0)
mx = float(np.abs(allpts).max()) * 1.15
ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx); ax.set_zlim(-mx, mx)

legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(0.5),
           markersize=5, label="Direction samples (reused across shells)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=cmap(0.5),
           markeredgecolor="black", markersize=7, label="Deterministic anchors"),
    Line2D([0], [0], color=cmap(0.3), alpha=0.6, lw=1.2,
           label="Sampled shells (ours, default range [%.1f, %.1f]r)" % RANGE_V10),
    Line2D([0], [0], color="#d62728", alpha=0.7, lw=1.2, ls="--",
           label="w/o-DA fixed range [0.5, 4.0]r"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
           markeredgecolor="darkred", markersize=11,
           label="Ground-truth cameras (100, all inside [0.5,4.0]r)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
           markersize=7, label="Mesh center"),
]
ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.0, 0.97),
          fontsize=8.5, framealpha=0.9)

ax.view_init(elev=22, azim=-58)
plt.tight_layout()
OUT = _args.out
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print("Saved:", OUT)
print("M=%d shells, S=%d dirs/shell, GT cameras=%d" % (M, S, len(GT_CENTERS)))
# sanity: confirm all GT distances lie inside v9 range [0.5,4.0]r
gt_d = np.linalg.norm(GT_CENTERS, axis=1) / R_MESH
print("GT dist/r  min=%.3f max=%.3f  (v9 range [0.5,4.0], v10 default range [%.1f,%.1f])"
      % (gt_d.min(), gt_d.max(), RANGE_V10[0], RANGE_V10[1]))
print("GT inside v9 [0.5,4.0]r: %d/%d" % (np.sum((gt_d >= 0.5) & (gt_d <= 4.0)),
                                           len(gt_d)))
