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
| `baselines/` | WAPR, Diff-DOPE, SDFR, FoundationPose, MegaPose, GoTrack, VGGT-, pi3-, Wid3R- and CAM3R-initialisation wrappers run on the exported frames |
| `tools/wsl/` | environments (PyTorch cu128) and run script for MegaPose (HappyPose) and GoTrack, whose released environments pin CUDA 11 |
| `tools/make_scflow2_data.py`, `tools/convert_scflow2_results.py`, `tools/scflow2_port/` | SCFlow2 (CVPR 2025) on the same frames and initial poses: data/config export, the CUDA 11 worker scripts, the Windows path patches, the lietorch / scikit-sparse drop-in shims and the result converter (see `tools/scflow2_port/README.md`) |
| `tools/export_revision_results.py`, `tools/export_bop_results.py`, `tools/make_qualitative_figures.py` | tables, macros and figures of the manuscript |
| `plot_sampling_space.py`, `tools/make_response_figures.py` | shell-layout figure of the Supplementary Material (Fig. S1) and the figures of the response letter |
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
python plot_sampling_space.py                             # Supplementary Fig. S1: the shell layout on the dragon (default range [0.4, 6.0] r)
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
# the proposed estimate (results.jsonl) as the starting pose of other refiners, and Diff-DOPE with its RGB term
python baselines/run_wapr_bop.py --root bop6dof_results/lmo --mode refine-from-ours   # also lm, tless
python baselines/run_sdfr.py --bop lmo --from-ours                                    # also lm, tless
python baselines/run_diffdope_bop.py --root bop6dof_results/lmo --from-ours           # also lm, tless
python baselines/run_diffdope_bop.py --root bop6dof_results/lmo --rgb                 # LM-O and LM (vertex colours)
python tools/analyze_fromours.py                                                      # Supplementary Table S11 and its macros
# hemisphere-agnostic seed ablation on the 20 views where WAPR's estimator fails (1/8 budget)
python tools/run_prior_ablation.py --method ours_v10_fast_ref                         # default seed rule
python tools/run_prior_ablation.py --method ours_v10_fast_sym                         # no elevation priors, symmetric bins/anchors
python tools/analyze_prior_ablation.py
# re-seeded hypotheses: the lowest-loss pose over the search and the four feed-forward sources, rotated about the
# principal axes of the mesh (180 deg; +-90 deg), each refined by stages II-IV
python bop_full6dof_eval.py --dataset lmo --from-export --init-from flip180 --out bop6dof_results/lmo   # also lm, tless
python bop_full6dof_eval.py --dataset lmo --from-export --init-from flip90 --out bop6dof_results/lmo
python tools/write_msf_selection.py                       # selection over all hypotheses -> results_msf_select.jsonl
python tools/icp_polish.py --src results_msf_select.jsonl --out results_msf_icp.jsonl   # training-free point-to-plane depth ICP (CPU)
python tools/icp_fitness_select.py                 # ICP on every hypothesis, then keep the best depth fit (results_msf_fit_sel.jsonl)
python tools/analyze_multisource.py                # fitness row of the multi-source table and its macros
python tools/dense_hyp.py --workers 8              # 252 analytic rotations; results in benchmark_results/dense_hyp_results.json (response letter only, not a manuscript table)
python bop_full6dof_eval.py --dataset lmo --from-export --init-from msf --size 384 --out-suffix _s384 --out bop6dof_results/lmo   # stages II-IV again at a 384^2 crop
python baselines/run_wapr_bop.py --root bop6dof_results/lmo --mode refine-from-ours --ours-file results_msf_select.jsonl --out-name baseline_wapr_from_msf.jsonl
# MegaPose and GoTrack (Linux / WSL 2 Ubuntu, PyTorch cu128): environments, then every protocol on both samples
bash tools/wsl/setup_envs.sh
bash tools/wsl/run_baselines.sh
# FoundationPose's refiner with 20 instead of 5 iterations; second sample; 1/8 budget; paired wide-offset comparison
python baselines/run_foundationpose_bop.py --root bop6dof_results/lmo --mode refine --bins 60-90,90-180 --iters 20
python bop_full6dof_eval.py --dataset lmo --objects 1,5,6,8,9,10,11,12 --frames-per-object 5 --seed 2027 --exclude-from bop6dof_results/lmo --out bop6dof_results_second/lmo   # LM, T-LESS: 4 frames per object
python bop_full6dof_eval.py --dataset lmo --from-export --out bop6dof_results_fast/lmo --budget 1/8
python tools/analyze_replication.py; python tools/analyze_wide_bin.py; python tools/analyze_fp_iterations.py
python tools/analyze_fast.py; python tools/analyze_residual_failures.py; python tools/analyze_failure_modes.py
```

## Protocol notes

* No ground-truth quantity enters the estimator: `VAL_STRICT_NO_GT=1` raises if any stage receives the
  ground-truth pose, and `tools/test_no_gt_leak.py` verifies bit-identical outputs under GT-label permutation.
* The depth channel is metric view depth; the distance hint is `median depth + kappa * r` with `kappa` calibrated
  from 32 random renders of the mesh (no observation involved).
* Runtimes in the tables were measured on a shared GPU; `stage_timing.json` holds CUDA-synchronised per-stage times.

See `REPRODUCTION_LOG.md` for the exact command history and software versions used for the submitted numbers.
