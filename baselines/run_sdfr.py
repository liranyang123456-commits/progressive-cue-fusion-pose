#!/usr/bin/env python3
"""
SDFR (Pang et al., IEEE T-RO 2025) on the BOP protocol and on the synthetic Stanford benchmark.

SDFR encodes the known mesh in a signed-distance network trained once per object from the CAD model alone and
refines a pose by minimising |SDF| of the observed points mapped into the object frame (no rendering, no training
data). The released code is used unchanged (external/SDFR: init_SDFR's network and training setting -- OctreeSDF,
5 LODs, 40 epochs -- and SDFR_adam's standard mode: 5000 observed points, farthest-point subsampling to 150, 100
Adam steps, rotation initialised from T0 and translation from the centroid of the observed points). The Windows /
torch 2.11 port replaces only infrastructure (tools/sdfr_port/README.md): a pytorch3d stand-in for so3_exp/log_map
and farthest-point sampling, and libigl's signed distance for the mesh2sdf CUDA kernel that generates the SDF
training samples; tools/sdfr_port/verify_port.py checks all three numerically.

The observed points are the masked depth pixels back-projected with K. SDFR's network lives in the frame of the
mesh re-centred at its bounding-box centre c, so its translation refers to c; it is mapped back to the model frame
by t = 1000 t_sdfr - R c (mm).

Run in the `sdfr` env:
  python baselines/run_sdfr.py --bop lmo            # all four T0 bins -> bop6dof_results/lmo/baseline_sdfr.jsonl
  python baselines/run_sdfr.py --bop lmo --from-ours  # from the proposed method's estimate -> baseline_sdfr_from_ours.jsonl
  python baselines/run_sdfr.py --synth bunny        # fixed initial camera -> benchmark_runs/bunny_100gt/sdfr/
  add --sanity to start from the ground-truth rotation (convention check)
"""
from __future__ import annotations

import os

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SDFR_DIR = ROOT / "external" / "SDFR"
WORK = ROOT / "external" / "sdfr_work"
sys.path.insert(0, str(ROOT / "tools" / "sdfr_port"))  # pytorch3d stand-in
sys.path.insert(0, str(SDFR_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))

BOP = {"lm": (Path(os.environ.get("BOP_ROOT", "data/bop")) / "lm", "models", "models_eval"), "lmo": (Path(os.environ.get("BOP_ROOT", "data/bop")) / "lm", "models", "models_eval"),
       "tless": (Path(os.environ.get("BOP_ROOT", "data/bop")) / "tless", "models_cad", "models_eval")}  # LM-O uses the (identical) LM models


def sdfr_api():
    import multiprocessing as mp
    if not getattr(mp.set_start_method, "_sdfr_port", False):  # lib.trainer sets it on every Trainer(); allow repeats
        _orig = mp.set_start_method

        def _set_start_method(method, force=False):
            try:
                _orig(method, force=True)
            except RuntimeError:
                pass
        _set_start_method._sdfr_port = True
        mp.set_start_method = _set_start_method
    import lib  # noqa: F401  (SDFR's package)
    sys.modules.setdefault("lib.rendering", types.ModuleType("lib.rendering"))  # pyrender dataset tools, unused here
    from lib.options import parse_options
    from lib.trainer import Trainer
    from sdfr import SDFR_adam, get_sdf_net
    return parse_options, Trainer, get_sdf_net, SDFR_adam


def prepare_model(ply: Path, key: str):
    """init_SDFR of run_sdfr.py for one mesh: OBJ copy, bounding-box centre / scale, trained OctreeSDF network."""
    import trimesh
    parse_options, Trainer, get_sdf_net, _ = sdfr_api()
    d = WORK / key
    d.mkdir(parents=True, exist_ok=True)
    obj = d / "model.obj"
    if not obj.exists():
        trimesh.load(str(ply), force="mesh", process=False).export(str(obj))
    V = np.asarray(trimesh.load(str(obj), force="mesh", process=False).vertices, dtype=np.float64)
    c = (V.max(0) + V.min(0)) / 2.0
    s = 1.0 / float(np.sqrt(((V - c) ** 2).sum(1).max()))
    args = parse_options(return_parser=True).parse_args([])
    args.gpu = 0
    args.net, args.num_lods = "OctreeSDF", 5
    if not (d / "model.pth").exists():
        args.dataset_path, args.epochs, args.model_path, args.exp_name, args.logs = str(obj), 40, str(d), "model", str(d / "logs")
        t0 = time.time()
        Trainer(args, "").train()
        (d / "train_time_s.txt").write_text(f"{time.time() - t0:.1f}", encoding="utf-8")
    return get_sdf_net(args, str(d / "model.pth")), s, c


def backproject(depth_mm: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    v, u = np.nonzero((mask > 0) & (depth_mm > 0))
    z = depth_mm[v, u].astype(np.float64)
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)


def refine(net, SDFR_adam, pts_mm: np.ndarray, R0, s: float, c_mm: np.ndarray, seed: int):
    import torch
    rng = np.random.default_rng(seed)
    trg = torch.from_numpy((pts_mm[rng.choice(len(pts_mm), 5000, replace=True)] / 1000.0).astype(np.float32))[None]
    init = torch.eye(4, dtype=torch.float32)[None]
    init[0, :3, :3] = torch.from_numpy(np.asarray(R0, dtype=np.float32))
    R, t, _ = SDFR_adam(net, trg, s, init, sample_farthest=True)
    R = R[0].numpy().astype(np.float64)
    return R, 1000.0 * t[0].numpy().astype(np.float64) - R @ c_mm


def load_ours(res_dir: Path) -> dict:
    ours = {}
    for line in (res_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r0 = json.loads(line)
            ours[r0["tag"]] = r0
        except Exception:
            pass
    return ours


def run_bop(ds: str, sanity: bool, max_frames: int, from_ours: bool = False, res_root: str = "bop6dof_results") -> None:
    import torch
    from bop_metrics import compute_metrics, eval_points_trimesh, load_models_info, symmetry_transforms
    _, _, _, SDFR_adam = sdfr_api()
    root, models, models_eval = BOP[ds]
    res_dir = ROOT / res_root / ds
    if from_ours:
        out = res_dir / "baseline_sdfr_from_ours.jsonl"
        ours = load_ours(res_dir)
    else:
        out = res_dir / ("baseline_sdfr_sanity.jsonl" if sanity else "baseline_sdfr.jsonl")
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line); done.add((r["tag"], r["bin"]))
            except Exception:
                pass
    frames = sorted(d for d in res_dir.iterdir() if d.is_dir() and (d / "protocol.json").exists() and (d / "observation.npz").exists())
    if max_frames > 0:
        frames = frames[:max_frames]
    nets, pts_cache, info_cache = {}, {}, {}
    for fdir in frames:
        proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
        obj = int(proto["obj_id"])
        key = f"{'lm' if ds in ('lm', 'lmo') else ds}_obj{obj:02d}"
        if key not in nets:
            nets[key] = prepare_model(root / models_eval / f"obj_{obj:06d}.ply", key)
            pts_cache[key] = eval_points_trimesh(root / models_eval, root / models, obj)
            info_cache[key] = load_models_info(root / models)[str(obj)]
        net, s, c = nets[key]
        obs = np.load(fdir / "observation.npz")
        K = obs["K"].astype(np.float64)
        pts = backproject(obs["depth"], obs["mask"], K)
        R_gt, t_gt = np.array(proto["gt_R"]), np.array(proto["gt_t"])
        if from_ours:
            r0 = ours.get(fdir.name)
            if r0 is None:
                print(f"[skip] {fdir.name}: no estimate of the proposed method")
                continue
            inits = [{"bin": "ours", "R0": r0["R_est"], "t0": r0["t_est"], "rot_err_deg": float(r0["e_R_deg"])}]
        else:
            inits = [{"bin": "gt", "R0": R_gt.tolist(), "rot_err_deg": 0.0}] if sanity else proto["inits"]
        for k, ini in enumerate(inits):
            if (fdir.name, ini["bin"]) in done:
                continue
            torch.cuda.synchronize(); t0 = time.perf_counter()
            no_points = len(pts) < 10  # no valid depth inside the mask (sensor holes): the refiner returns its input
            if no_points:
                R, t = np.array(ini["R0"], dtype=np.float64), np.array(ini.get("t0", t_gt), dtype=np.float64)
            else:
                R, t = refine(net, SDFR_adam, pts, ini["R0"], s, c, seed=1000 * k + obj)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            met = compute_metrics(R, t, R_gt, t_gt, K, pts_cache[key], info_cache[key], symmetry_transforms(info_cache[key]))
            rec = {"tag": fdir.name, "dataset": ds, "scene_id": proto["scene_id"], "im_id": proto["im_id"], "obj_id": obj,
                   "method": "ours_then_sdfr" if from_ours else "sdfr", "bin": ini["bin"], "init_rot_err_deg": ini.get("rot_err_deg"), "n_obs_points": int(len(pts)),
                   "no_observed_points": bool(no_points),
                   "R_est": R.tolist(), "t_est": t.tolist(), "t_total_s": dt, **met}
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{ds} {fdir.name} {ini['bin']:>6s}] e_R={met['e_R_deg']:6.1f}  e_t={met['e_t_mm']:6.1f} mm  ADD(-S)<0.1d={met['add_lt_0.1d']}  {dt:.2f}s", flush=True)


def run_synth(mesh: str, sanity: bool, max_frames: int) -> None:
    import torch
    from synth_common import SummaryWriter, synth_metrics
    _, _, _, SDFR_adam = sdfr_api()
    root = ROOT / "synth_frames" / mesh
    frames = sorted(d for d in root.iterdir() if d.is_dir() and (d / "protocol.json").exists())
    if max_frames > 0:
        frames = frames[:max_frames]
    proto0 = json.loads((frames[0] / "protocol.json").read_text(encoding="utf-8"))
    net, s, c = prepare_model(Path(proto0["mesh_path"]), f"synth_{mesh}")
    out = SummaryWriter(mesh, "sdfr_sanity" if sanity else "sdfr")
    for i, fdir in enumerate(frames):
        if fdir.name in out.done:
            continue
        proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
        obs = np.load(fdir / "observation.npz")
        pts = backproject(obs["depth"], obs["mask"], obs["K"].astype(np.float64))
        R0 = proto["gt_R"] if sanity else proto["inits"][0]["R0"]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        R, t = refine(net, SDFR_adam, pts, R0, s, c, seed=i)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        met = synth_metrics(R, t, proto)
        out.write(fdir.name, met, dt, {"R_est": R.tolist(), "t_est_mm": t.tolist()})
        print(f"[{mesh} {fdir.name}] e_R={met['rot_err_deg']:6.1f}  |dd|/d={met['dist_err'] / max(1e-12, met['gt_dist']):.3f}  {dt:.2f}s", flush=True)
    print(f"[done] {out.csv}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", choices=list(BOP.keys()))
    ap.add_argument("--synth")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--from-ours", action="store_true",
                    help="BOP only: start from the proposed method's estimate in results.jsonl (no T0) instead of the T0 bins")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--res-root", default="bop6dof_results", help="results root holding the exported frame packages of --bop")
    args = ap.parse_args()
    if args.bop:
        run_bop(args.bop, args.sanity, args.max_frames, args.from_ours, args.res_root)
    if args.synth:
        run_synth(args.synth, args.sanity, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
