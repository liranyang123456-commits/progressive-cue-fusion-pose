# SDFR on Windows / torch 2.11 (RTX 5090)

SDFR (Pang et al., *Fast and Accurate 6-D Object Pose Refinement via Implicit Surface Optimization*, IEEE T-RO 41,
2025; https://github.com/pangbo1997/SDFR, commit d29ad33) is released for Linux (Python 3.8, a shell script that
builds two CUDA extensions, pytorch3d). It was run unchanged in its method and settings; only infrastructure was
replaced. `baselines/run_sdfr.py` drives it on the BOP protocol (`--bop lmo|lm|tless`, the four T0 bins) and on the
synthetic benchmark (`--synth <mesh>`, the fixed initial camera; `--sanity` starts from the ground-truth rotation).

## Setup
```
git clone https://github.com/pangbo1997/SDFR external/SDFR && git -C external/SDFR checkout d29ad33
git -C external/SDFR apply ../../tools/sdfr_port/sdfr_windows.patch    # items 2-5 below
conda create -n sdfr --clone <env with torch 2.11 + cu128>; pip install open3d trimesh libigl rtree polyscope
    plyfile scikit-image tinyobjloader tensorboard (tinyobjloader: --pre --only-binary :all:)
python tools/sdfr_port/verify_port.py
python baselines/run_sdfr.py --synth bunny --sanity    # convention check from the ground-truth rotation
```
`build_ext_windows.py` (optional) builds the two CUDA extensions under a non-UTF-8 Windows locale; with MSVC 2019
the build stops at torch's `compiled_autograd.h` (C2872), which is why item 2 exists.

## What is unchanged
* the SDF network and its training (`init_SDFR` in `run_sdfr.py`: OctreeSDF, 5 LODs, 40 epochs, all other
  options at the defaults of `lib/options.py`; one network per mesh, trained from the CAD model alone);
* the refinement (`SDFR_adam` in `sdfr.py`, standard mode): 5000 observed points sampled with replacement,
  farthest-point subsampling to 150, 100 Adam steps (lr 5e-2 rotation / 3e-3 translation), rotation initialised
  from T0 and translation from the centroid of the observed points.

## What was replaced (environment `sdfr` = Python 3.11, torch 2.11 + CUDA 12.8)
1. **pytorch3d** (no build for torch 2.11 / Windows): `tools/sdfr_port/pytorch3d/` provides the three functions
   SDFR calls, written after pytorch3d's own code: `so3_exp_map`, `so3_log_map` (same `cos_bound` clamp) and
   `ops.sample_farthest_points` (deterministic, starting at index 0 as pytorch3d's default). `se3_log_map` and
   `iterative_closest_point` are imported by `sdfr.py` but not used by `SDFR_adam`; they raise if called.
2. **mesh2sdf CUDA kernel** (training-sample SDF values): the extension does not compile with MSVC 2019 against
   torch 2.11 headers (`error C2872: 'std': ambiguous symbol`), so `lib/torchgp/compute_sdf.py` falls back to
   libigl's signed distance (fast winding number sign) when `mesh2sdf` is missing. The SDF of a mesh is a property
   of the geometry; the substitute only computes it differently.
3. Hard imports made optional: `mesh2sdf` (`lib/datasets/MeshDataset.py`), `SOL_NGLOD` (`lib/models/__init__.py`,
   a model not used by OctreeSDF) and `sol_nglod.aabb` (`lib/tracer/SphereTracer.py`, used only to render log
   images, which training does not call). `lib.rendering` (pyrender / dataset preparation) is stubbed by the driver.
4. `multiprocessing.set_start_method` is made repeatable so that several networks can be trained in one process.
5. Extension `setup.py` flags: `/std:c++17` / `-std=c++17` on Windows (torch 2.x headers need C++17); the
   extensions are optional after (2)-(3).

## Verification (`python tools/sdfr_port/verify_port.py`)
* `so3_exp_map` vs scipy `Rotation.from_rotvec`, 2000 random rotations up to 179 deg: max |dR| = 4.3e-8
* `so3_log_map` round trip vs scipy `as_rotvec`: max deviation 1.8e-7
* farthest point sampling: indices identical to a brute-force reference
* libigl vs trimesh (independent implementation) on 3000 near-surface points of LM object 1 (watertight):
  max |difference| of |SDF| 0.016 mm, sign agreement 100 %
* trained network (LM object 10): |SDF| of exact model-surface points at the ground-truth pose, median 0.07 mm

## Frame conventions
SDFR's network lives in the frame of the mesh re-centred at its bounding-box centre c and scaled to the unit sphere;
`SDFR_adam` maps observed points p (metres) as (p - t) R * s * 1000, so its translation t refers to c. The driver
returns the model-frame pose (R, 1000 t - R c) in mm, scored with the same metrics as every other method.
