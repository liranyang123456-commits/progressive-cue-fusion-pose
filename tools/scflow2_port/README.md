# Running SCFlow2 (CVPR 2025) on our BOP protocol

SCFlow2 (https://github.com/W-QY/SCFlow2) is pinned to PyTorch 2.0.1 / CUDA 11.7 (mmcv-full 1.7.2,
pointnet2 CUDA ops, PyTorch3D 0.7.4, lietorch, scikit-sparse). This directory contains everything
that was needed to run its released checkpoint on a second machine (RTX 2080 Ti, Windows 10,
Python 3.10, CUDA 11.7, MSVC 2019) with the frames, masks, intrinsics and perturbed initial
poses exported by our protocol.

| file | purpose |
|---|---|
| `../make_scflow2_data.py` | converts `bop6dof_results/<ds>/` (observation.npz + protocol.json) into the BOP-style layout SCFlow2 reads (`rgb/`, `depth/`, `mask_visib/`, `scene_camera/gt/gt_info.json`), writes the image lists, the per-bin `initial_poses/ours_bin*/` folders and the configs below |
| `configs/` | SCFlow2 configs: `ds_<ds>.py` (dataset, class lists, symmetries), `scflow2_<ds>.py` (model), `<ds>_bin<a>to<b>.py` (initial-pose bin) |
| `scflow2_step1..5.cmd` | environment build on the worker (Python 3.10 venv, torch 2.0.1+cu117, mmcv-full 1.7.2, pointnet2_ops, PyTorch3D, remaining pip deps with numpy 1.26 pinned); paths are the worker's `%USERPROFILE%` |
| `scflow2_run.cmd` | runs `test.py` for the 12 dataset x bin configs and writes one log per run |
| `patch_scflow2_windows.py` | path-handling patches (forward-slash joins, `posixpath`); no numerical code is touched |
| `lietorch_shim.py` | pure-PyTorch replacement of `lietorch.SE3` (exp/log/inv/compose/act/retr/adj); the compiled `lietorch_backends` crashes under MSVC. Install as `site-packages/lietorch/__init__.py`. Verified against `torch.linalg.matrix_exp` on random tangent vectors (max deviation 1.6e-15) |
| `sksparse_shim_cholmod.py` | replacement of `sksparse.cholmod.analyze_AAt` used by SCFlow2's grid smoother: SciPy SuperLU factorisation of the same normal equations. Install as `site-packages/sksparse/cholmod.py` |
| `../convert_scflow2_results.py` | maps SCFlow2's `scene_gt.json` outputs back to our frame tags and writes `bop6dof_results/<ds>/baseline_scflow2.jsonl` with the same symmetry-aware metrics as every other row |

Order: `make_scflow2_data.py` -> copy data + configs + checkpoint to the worker -> `scflow2_step*.cmd` ->
install the two shims -> `patch_scflow2_windows.py <checkout>` -> `scflow2_run.cmd` -> copy
`results/ours/` back -> `convert_scflow2_results.py --results <dir>` -> `tools/export_bop_results.py`.

SCFlow2 receives, per frame, RGB-D, K, the CAD model and the perturbed pose T0; as in the authors' test
configuration the crop is centred on the ground-truth 2-D box (`crop_bbox_field='gt_bboxes'`), i.e. the same
2-D information the other methods obtain from the given mask. Its runtime (about 1 s/frame on the 2080 Ti)
is not comparable with the rows measured on the RTX 5090.
