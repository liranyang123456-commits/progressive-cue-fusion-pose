# Known-geometry camera pose estimation via progressive cue fusion -- code and results

Code and results accompanying the revised manuscript *Known Geometry Camera Pose Refinement via Progressive Cue Fusion* (Pattern Recognition, PR-D-26-09325).
Everything needed to regenerate every table and figure of the paper from the raw per-view
outputs is included; the GPU runs themselves can be reproduced with the commands below.

## Layout

| path | content |
|---|---|
| `Exp_Camera_Optim_RT_Val_Beam_Search12291_Standford122.py` | the pipeline (stage I RandMultiR global search, stage II depth-guided distance alignment, stage III LocalBeam, stage IV VisualSelect) and all baseline modes, selected through `VAL_*` environment variables |
| `run_benchmark_compare.py` | method configurations (`METHODS`), mesh registry (`DATASETS`), one-view resume logic |
| `run_revision_queue.py` | ordered job queue used for the revision runs (strict no-ground-truth protocol) |
| `tools/test_no_gt_leak.py` | automated ground-truth-leak test: estimates must be bit-identical under a permutation of GT labels with fixed observations |
| `bop_full6dof_eval.py`, `bop_metrics.py` | full 6-DoF protocol on BOP LM / LM-O / T-LESS (virtual centred camera, roll bank, symmetry-aware ADD(-S) / MSSD / MSPD), perturbed-pose export for refiners, stages II-IV from an initial pose |
| `baselines/` | WAPR, Diff-DOPE, VGGT- and pi3-initialisation wrappers run on the exported frames |
| `tools/make_scflow2_data.py`, `tools/convert_scflow2_results.py`, `tools/scflow2_port/` | SCFlow2 (CVPR 2025) on the same frames and initial poses: data/config export, the CUDA 11 worker scripts, the Windows path patches, the lietorch / scikit-sparse drop-in shims and the result converter (see `tools/scflow2_port/README.md`) |
| `tools/export_revision_results.py`, `tools/export_bop_results.py`, `tools/make_qualitative_figures.py` | tables, macros and figures of the manuscript |
| `results/` | per-view metrics (`pose_metrics_summary.csv`), stage timings, BOP `*.jsonl`, leak-test reports |

## Environment

* Python 3.10+, PyTorch >= 2.4 with CUDA, `nvdiffrast`, `open3d`, `opencv-python`, `numpy`, `scipy`, `matplotlib`
  (`pip install -r requirements.txt`; nvdiffrast is compiled on first import and needs a CUDA toolkit + C++ compiler).
* Set `STANFORD_ROOT` to a directory with `bun_zipper.ply` and `xyzrgb_dragon.ply`
  (Stanford 3D Scanning Repository); `python tools/download_stanford_meshes.py` fetches and decimates
  Armadillo / Happy Buddha / Lucy into `benchmark_assets/`.
* Set `BOP_ROOT` to a BOP layout containing `lm`, `lmo`, `tless` (`models`, `models_eval`, `test`).
* Set `PIPELINE_PYTHON` to the interpreter that has the dependencies if it is not the default `python`.

## Reproducing the synthetic benchmark

```bash
export VAL_STRICT_NO_GT=1
python run_revision_queue.py --plan core --workers 2      # bunny / dragon N=100, all methods; ablations; new meshes
python tools/derive_geom_only.py --run-root benchmark_runs/bunny_100gt   # Geom-only selector (offline, same candidates)
python run_benchmark_compare.py --dataset bunny --num-gt 100 --methods ours_v10_fast --max-views 30   # 1/8 budget (M=20, S=256, K=8, B=24, L=16); also dragon
python tools/export_revision_results.py                   # tables/*.tex, results_macros.tex, binned-SR figure
python tools/make_qualitative_figures.py
python tools/test_no_gt_leak.py --dataset bunny --methods ours_v10,grid_adam,rand_restart_adam --num-gt 3 --legacy
# recent methods on exactly the same observations (640x480 RGB, metric depth, silhouette; mesh scaled to r = 100 mm)
python tools/export_synth_frames.py                              # synth_frames/<mesh>/gt_XX/{observation.npz, protocol.json}
python baselines/run_wapr_synth.py --mesh bunny --mode full      # WAPR environment; --mode refine, --sanity (from the ground truth)
python baselines/run_diffdope_synth.py --mesh bunny              # Diff-DOPE from the fixed initial camera; --sanity
```

## Reproducing the BOP evaluation

```bash
python bop_full6dof_eval.py --selftest
python bop_full6dof_eval.py --dataset lmo --objects 1,5,6,8,9,10,11,12 --frames-per-object 5 --out bop6dof_results/lmo
python bop_full6dof_eval.py --dataset lmo --export-only ... # writes observation.npz + protocol.json (T0 bins) for the refiners
python bop_full6dof_eval.py --dataset lmo --init-from bins  # stages II-IV from the same T0 bins
python baselines/run_wapr_bop.py --root bop6dof_results/lmo --mode refine   # WAPR environment
python baselines/run_diffdope_bop.py --root bop6dof_results/lmo              # Diff-DOPE environment
python tools/render_templates_for_fm.py --root bop6dof_results/lmo
python baselines/run_vggt_init.py --root bop6dof_results/lmo                # VGGT environment
python bop_full6dof_eval.py --dataset lmo --init-from vggt
python baselines/run_pi3_init.py --root bop6dof_results/lmo                 # pi3 environment
python bop_full6dof_eval.py --dataset lmo --init-from pi3
python baselines/run_wid3r_init.py --root bop6dof_results/lmo               # wapr environment + external/Wid3R
python bop_full6dof_eval.py --dataset lmo --init-from wid3r
python baselines/run_cam3r_init.py --root bop6dof_results/lmo               # cam3r environment + external/cam3r
python bop_full6dof_eval.py --dataset lmo --init-from cam3r
python bop_full6dof_eval.py --dataset lmo --from-export --occlusion-aware --out-suffix _occ   # ablation (margin r/4); also lm, tless
python tools/compare_occlusion_aware.py
python baselines/run_sdfr.py --bop lmo                                      # sdfr environment, tools/sdfr_port/README.md; also lm, tless
python baselines/run_sdfr.py --synth bunny [--sanity]                       # SDFR on the synthetic benchmark (all five meshes)
python tools/make_scflow2_data.py                                           # SCFlow2 input packages + configs (run SCFlow2 on a CUDA 11 machine, see tools/scflow2_port/README.md)
python tools/convert_scflow2_results.py --results external/scflow2_results
python tools/export_bop_results.py
```

## Protocol notes

* No ground-truth quantity enters the estimator: `VAL_STRICT_NO_GT=1` raises if any stage receives the
  ground-truth pose, and `tools/test_no_gt_leak.py` verifies bit-identical outputs under GT-label permutation.
* The depth channel is metric view depth; the distance hint is `median depth + kappa * r` with `kappa` calibrated
  from 32 random renders of the mesh (no observation involved).
* Runtimes in the tables were measured on a shared GPU; `stage_timing.json` holds CUDA-synchronised per-stage times.

See `REPRODUCTION_LOG.md` for the exact command history and software versions used for the submitted numbers.
