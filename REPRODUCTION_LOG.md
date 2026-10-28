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
# tables / figures
python tools/derive_geom_only.py --run-root benchmark_runs/<mesh>_<N>gt
python tools/export_revision_results.py
python tools/export_bop_results.py
python tools/make_qualitative_figures.py
```

## Run status

(filled in as runs complete; see `benchmark_results/revision_summary.json` and `bop_summary.json`)
