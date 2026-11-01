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
