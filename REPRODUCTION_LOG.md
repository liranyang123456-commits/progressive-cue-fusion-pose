# Reproduction log -- revision of PR-D-26-09325

All numbers in the revised manuscript come from runs made after the three protocol corrections
described in the response letter (strict no-ground-truth mode, metric depth channel, text/code
alignment). Runs made with the submitted code are archived under `benchmark_runs_legacy_leaky/`
and are not used anywhere in the revision.

## Software

| component | version |
|---|---|
| Python (control machine) | 3.12 (conda env `kaolin`) |
| PyTorch | 2.8.0+cu128 |
| nvdiffrast | 0.4.0 |
| Open3D | 0.19.0 |
| OpenCV | 4.10.0 |
| NumPy | 1.26.4 |
| WAPR baseline env | Python 3.12, torch 2.11.0+cu128, `wapr` (pip), TensorRT engines built locally |
| Diff-DOPE | upstream repository, installed `--no-deps` into the `kaolin` env |
| VGGT | `facebook/VGGT-1B`, pip install from `facebookresearch/vggt` |
| pi3 | `yyfz233/Pi3` checkpoint, code from `yyfz/Pi3` (run in float16; the bf16 path requires flash-attention SDPA, unavailable on sm_120) |
| SCFlow2 | upstream `W-QY/SCFlow2` + released checkpoint, run on the GPU worker in a separate Python 3.10 venv: torch 2.0.1+cu117, mmcv-full 1.7.2, pointnet2_ops, PyTorch3D 0.7.4 (built), lietorch replaced by `tools/scflow2_port/lietorch_shim.py`, scikit-sparse by `tools/scflow2_port/sksparse_shim_cholmod.py`; see `tools/scflow2_port/README.md` |

Hardware: control machine RTX 5090 Laptop GPU (24 GB, shared with unrelated training jobs during the
runs -- wall-clock times in the tables are therefore upper bounds); GPU worker RTX 2080 Ti (11 GB,
Python 3.11.9, torch 2.4.1+cu118, nvdiffrast built from source with CUDA 11.7 / MSVC 14.29).
Observations (silhouette, depth, normals, RGB renders of every ground-truth view) were rendered once on
the control machine and copied to the worker (`tools/make_worker_bundle.py`), so every method of a
table sees bit-identical inputs regardless of where it ran.

## Protocol corrections (what changed with respect to the submitted code)

1. `VAL_STRICT_NO_GT=1` (default): the normal-map view-direction prior no longer receives the
   ground-truth rotation, LocalBeam no longer conditions on the ground-truth elevation, and
   `_assert_no_gt` raises if a ground-truth tensor reaches any scoring function.
   `tools/test_no_gt_leak.py` permutes the GT labels while keeping the observations fixed and requires
   bit-identical estimates (tolerance 1e-4 on dist / elev / azim).
2. The depth channel of the rasteriser was NDC z/w; it is now perspective-correct metric view depth.
   The distance hint of stage II is `median(depth) + kappa * r` with kappa calibrated from 32 random
   renders of the mesh (`[DepthHint] mesh centre offset kappa=...` in every run log).
3. Equations (4a-c), the per-shell sample count S=1024, the radius schedule and the nested channel
   sets in the text were aligned with the implementation.

## Command history

```
# observation caches + reference method (control machine)
set VAL_STRICT_NO_GT=1
python run_revision_queue.py --plan core --workers 2          # started 2026-10-28 12:49
python run_revision_queue.py --plan ours_rest --workers 1     # ours-family ablations + new meshes
# cheap baselines (GPU worker, identical cached observations)
python tools/make_worker_bundle.py --out G:/worker_bundle --zip G:/worker_bundle.zip
python worker_queue.py --root <worker>/pose_bench --python <venv>/python.exe
# leak test
python tools/test_no_gt_leak.py --dataset bunny  --methods ours_v10,grid_adam,rand_restart_adam --num-gt 3 --legacy
python tools/test_no_gt_leak.py --dataset dragon --methods ours_v10,grid_adam,rand_restart_adam --num-gt 3 --legacy
# BOP
python bop_full6dof_eval.py --selftest
python bop_full6dof_eval.py --dataset lmo   --objects 1,5,6,8,9,10,11,12   --frames-per-object 5 --export-only --out bop6dof_results/lmo
python bop_full6dof_eval.py --dataset lm    --objects 1,5,6,8,9,10,11,12   --frames-per-object 4 --export-only --out bop6dof_results/lm
python bop_full6dof_eval.py --dataset tless --objects 1,5,9,14,20,23,27,30 --frames-per-object 4 --export-only --out bop6dof_results/tless
python bop_full6dof_eval.py --dataset <ds> --from-export --out bop6dof_results/<ds>                      # proposed, stages I-IV
python bop_full6dof_eval.py --dataset <ds> --from-export --init-from bins --out bop6dof_results/<ds>     # proposed, stages II-IV from T0
python baselines/run_wapr_bop.py --root bop6dof_results/<ds> --mode refine ; --mode full
python baselines/run_diffdope_bop.py --root bop6dof_results/<ds>
python tools/render_templates_for_fm.py --root bop6dof_results/<ds>
python baselines/run_vggt_init.py --root bop6dof_results/<ds>
python bop_full6dof_eval.py --dataset <ds> --from-export --init-from vggt --out bop6dof_results/<ds>
python baselines/run_pi3_init.py --root bop6dof_results/<ds>
python bop_full6dof_eval.py --dataset <ds> --from-export --init-from pi3 --out bop6dof_results/<ds>
python bop_full6dof_eval.py --dataset lmo --from-export --occlusion-aware --out bop6dof_results/lmo       # ablation row
# reduced budget (1/8 of the proposal and beam budgets), first 30 views; bunny on the RTX 2080 Ti worker, dragon locally
python run_benchmark_compare.py --dataset bunny --num-gt 100 --methods ours_v10_fast --max-views 30
python run_benchmark_compare.py --dataset dragon --num-gt 100 --methods ours_v10_fast --max-views 30
# WAPR and Diff-DOPE on the synthetic benchmark (same observations as every method of Tables 1-2)
python tools/export_synth_frames.py                              # GL world-to-view -> OpenCV, mesh re-centred and scaled to r = 100 mm, fixed initial camera as T0
python baselines/run_wapr_synth.py --mesh <mesh> --sanity        # from the ground truth (convention check)
python baselines/run_wapr_synth.py --mesh <mesh> --mode full     # 12 hypotheses, no initial pose
python baselines/run_wapr_synth.py --mesh <mesh> --mode refine   # from the fixed initial camera
python baselines/run_diffdope_synth.py --mesh <mesh>             # from the fixed initial camera (published hyper-parameters)
python baselines/run_diffdope_synth.py --mesh <mesh> --sanity    # from the ground truth
# occlusion-aware silhouette term (margin r/4 after the OCC_MESH_R_MM fix), all three datasets, then the paired test
python bop_full6dof_eval.py --dataset <ds> --from-export --occlusion-aware --out-suffix _occ --out bop6dof_results/<ds>
python tools/compare_occlusion_aware.py
# Wid3R / CAM3R initialisation (R1-2), then stages II-IV
python baselines/run_wid3r_init.py --root bop6dof_results/<ds>     # wapr env, 518x336 as in Wid3R's evaluation code
python baselines/run_cam3r_init.py --root bop6dof_results/<ds>     # cam3r env (PYTHONIOENCODING=utf-8), most confident (crop, template) pair
python bop_full6dof_eval.py --dataset <ds> --from-export --init-from wid3r|cam3r --out bop6dof_results/<ds>
# SDFR (T-RO 2025) on BOP and on the synthetic benchmark (sdfr env; tools/sdfr_port/README.md)
python tools/sdfr_port/verify_port.py
python baselines/run_sdfr.py --bop <ds>
python baselines/run_sdfr.py --synth <mesh> --sanity && python baselines/run_sdfr.py --synth <mesh>
# SCFlow2 (CUDA 11 worker; tools/scflow2_port/README.md)
python tools/make_scflow2_data.py                                   # -> external/scflow2_data (BOP layout, image lists, initial_poses/ours_bin*, configs)
#   worker: scflow2_step1..5.cmd, install the two shims, patch_scflow2_windows.py <checkout>, scflow2_run.cmd
python tools/convert_scflow2_results.py --results external/scflow2_results
# tables / figures
python tools/derive_geom_only.py --run-root benchmark_runs/<mesh>_<N>gt
python tools/export_revision_results.py
python tools/export_bop_results.py
python tools/make_qualitative_figures.py
```

## Run status

(see `benchmark_results/revision_summary.json` and `bop_summary.json`)

* SCFlow2: 12 configs (3 datasets x 4 bins), all exit 0; 416 refined poses; ~1.1 s/frame on the 2080 Ti.
  One T-LESS run initially failed because the DINOv2 hub download was interrupted (`RemoteDisconnected`);
  it was re-run with `TORCH_HOME` pointing at the pre-populated hub cache. The time column of SCFlow2 is
  therefore measured on a different GPU than all other BOP rows and is marked as not comparable.
* WAPR / Diff-DOPE on the synthetic benchmark (11-02): the exporter checks every view by projecting the mesh with the
  ground-truth pose into the silhouette (>= 96% of the vertices inside) and verifies the reconstructed initial camera
  against the saved one; mask = (alpha > 0.5) & (depth > 0) so that silhouette and depth agree at anti-aliased edges.
  From the ground truth WAPR stays within 10 deg on (nearly) every view; Diff-DOPE is unstable on close-ups (d/r < 2),
  independently of the chosen metric scale (r = 100 / 300 / 1000 mm tried on 8 Bunny views), so r = 100 mm is used for both.
* pi3 initialisation: LM-O / LM / T-LESS, 12 rendered templates + real crop, float16; refined with stages II-IV
  (`results_init_pi3.jsonl`).
* Refiners started from the proposed estimate (11-07): WAPR's refiner, SDFR and Diff-DOPE read the stage I-IV estimate
  from `results.jsonl` in place of T0 (`--mode refine-from-ours` / `--from-ours`); SR@10 sym. over the 104 frames
  62 / 42 / 42% against 22 / 7 / 8% from the 90-180 deg bin and 35% for the estimate alone. Diff-DOPE with its RGB term
  (`--rgb`, vertex colours of the LM / LM-O models) from the same T0 bins: lower in every bin (4 of 288 paired runs
  gained, 15 lost). Summary and macros: `tools/analyze_fromours.py` -> `benchmark_results/bop_fromours_summary.json`.
* Seed priors and the upper hemisphere (11-07): the 20 views on which WAPR's estimator fails, at the 1/8 budget,
  with the default seed rule (`ours_v10_fast_ref`) and without the two elevation-related priors and with seed bins /
  anchors symmetric about the equator (`ours_v10_fast_sym`): 19 / 19 solved (median 0.35 / 0.43 deg; full budget 19,
  WAPR 0), no view changes outcome; the failure in all runs is the flipped Happy Buddha view gt_05.
* Controls added while aligning the text with the released code (11-11).
  (a) Auxiliary terms. The archived full-budget synthetic runs (`ours_v10`) contained an appearance term in the order of
  the stage-I seeds (`VAL_GLOBAL_RGB_RERANK=1`) and, on beam levels whose candidates have a mean elevation >= 45 deg,
  depth / normal weights raised by 2.5 / 1.5 plus an appearance term (`VAL_LOCAL_BEAM_HIGH_ELEV_AUX=1`). Both are now
  off by default; `ours_v10_aux` / `ours_v10_fast_aux` switch them on. At the 1/8 budget without them
  (`ours_v10_fast_clean`, all 350 views): 348 within 10 deg (full budget with the terms: 349), median e_R
  0.39 deg, 0 wrong shells; the two misses (Happy Buddha gt_05, gt_37) are also missed with the terms
  at that budget. With the terms on 51 views (first 10 per mesh and the views the clean run misses): 49 against 49
  solved, the same views; 44 identical estimates, largest difference 0.14 deg.
  (b) Stage I only (`tools/derive_stage1_only.py`): the lowest-loss proposal of the full-budget run is within 10 deg in
  319 of 350 views; at the 1/8 budget in 252.
  (c) Retained seeds and beam width, on the 98 views whose lowest-loss proposal is outside 10 deg at the 1/8 budget:
  all K=8 seeds with the width-24 beams 96, with a width-one walk (`ours_v10_fast_w1`) 96; only the lowest-loss seed
  (`VAL_LOCAL_BEST_SEED_ONLY=1`) with the beam (`ours_v10_fast_k1`) 90, with the walk (`ours_v10_fast_k1w1`) 89.
  Summary: `tools/analyze_aux_control.py` -> `benchmark_results/aux_control_summary.json`.
  (d) Baselines. The DR baselines, Rand-TopK and Greedy-BT ran under the v9 profile (s_img = 1, fixed 36 shells
  0.5r-4r) and no multi-start refinement was executed: RandRestart+Adam is one Adam run of 400 iterations from the
  Rand-TopK pose. It starts within 10 deg in 137 of 350 views and ends within 10 deg in 89 (56 lost, 8 gained);
  Grid+Adam 70 -> 41 (`tools/analyze_adam_start_end.py`).
  (e) The optional Adam polish (`VAL_LOCAL_BEAM_FINAL_DR_ENABLE`) was off in every reported run; the paper no longer
  describes it.
  (f) BOP seeding, re-weighting and full-budget 2x2 controls (11-12). All full-budget BOP runs on both disjoint samples
  (`bop6dof_results`, `bop6dof_results_second`; 104 + 104 = 208 frames), the 104-frame 1/8-budget run
  (`bop6dof_results_fast`) and every downstream BOP step (6 `refine-from-ours` refiners, `flip180`/`flip90`,
  `results_msf_select.jsonl`, `results_init_msf_s384.jsonl`, `refine-from-msf`, `icp_polish.py`, `icp_fitness_select.py`,
  `dense_hyp.py`) were recomputed with per-frame seeded stage-I proposals (`seed_frame(tag, seed)`) and uniform beam
  weights (released default): Sample 1 SR@10 sym. 35.6% (37/104; +FM 57.7%, +re-seeded 69.2%, +ICP 78.8%, +fitness
  86.5%; 6 refiners from ours 44-66%), Sample 2 29.8% (31/104; both samples 32.7% -> 84.1% fitness).
  On every third frame (36 frames) at the 1/8 budget, seeded: 15 within 10 deg sym. with uniform weights, 15 with
  `--high-elev-weights`, 35 of 36 estimates identical (`tools/analyze_bop_reweight_control.py`).
  Full-budget 2x2 controls (`tools/analyze_full_controls.py` -> `benchmark_results/full_controls_summary.json`):
  - Synthetic (the 31 views where Stage I > 10 deg): all K=16 seeds + B=48 beam (`ours_v10_clean`) 30/31, K=16 + B=1
    walk (`ours_v10_w1`) 29/31 (p=1.00), K=1 + B=48 (`ours_v10_k1`) 29/31, K=1 + B=1 (`ours_v10_k1w1`) 29/31.
  - BOP (all 208 frames of both samples, identical cached Stage-I proposals): K=8 + B=48 68/208 (32.7%), K=8 + B=1
    walk (`results_w1.jsonl`) 67/208 (32.2%, +1/-0, p=1.00; 183/208 identical to 1e-6 deg; local time 46s -> 5s),
    K=1 + B=48 (`results_k1.jsonl`) 57/208 (27.4%, +15/-4 vs K=8, exact McNemar p=0.019), K=1 + B=1
    (`results_k1_w1.jsonl`) 57/208 (27.4%, +15/-5 vs K=8 walk, p=0.041).
