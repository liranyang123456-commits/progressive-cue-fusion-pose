#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert SCFlow2 outputs (test.py --format-only writes <save_dir>/<scene>/scene_gt.json with the refined poses) into our
per-frame jsonl records with the shared metrics, one record per (frame, initial-pose bin):

  python tools/convert_scflow2_results.py --results external/scflow2_results --data external/scflow2_data

Expected layout (as produced on the worker):  <results>/<ds>_bin<15to30|30to60|60to90|90to180>/<scene:06d>/scene_gt.json
Writes bop6dof_results/<ds>/baseline_scflow2.jsonl (method "scflow2_refine").
"""
from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_metrics import compute_metrics, eval_points_trimesh, load_models_info, symmetry_transforms  # noqa: E402

BOP = Path(os.environ.get("BOP_ROOT", "data/bop"))
MODELS = {"lm": ("models", "models_eval"), "lmo": ("models", "models_eval"), "tless": ("models_cad", "models_eval")}
BINS = {"15to30": "15-30", "30to60": "30-60", "60to90": "60-90", "90to180": "90-180"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", default=str(ROOT / "external" / "scflow2_data"))
    ap.add_argument("--datasets", default="lmo,lm,tless")
    args = ap.parse_args()
    res_root, data_root = Path(args.results), Path(args.data)
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        tag_index = json.loads((data_root / "data" / ds / "tag_index.json").read_text(encoding="utf-8"))
        mdir, edir = MODELS[ds]
        info_all = load_models_info(BOP / ds / mdir)
        pts_cache, sym_cache = {}, {}
        out = ROOT / "bop6dof_results" / ds / "baseline_scflow2.jsonl"
        recs = []
        for key, bname in BINS.items():
            run = res_root / f"{ds}_bin{key}"
            if not run.exists():
                print(f"[{ds}] missing {run}")
                continue
            for sdir in sorted(run.iterdir()):
                gt_p = sdir / "scene_gt.json"
                if not sdir.is_dir() or not gt_p.exists():
                    continue
                scene = int(sdir.name)
                preds = json.loads(gt_p.read_text(encoding="utf-8"))
                for im_s, lst in preds.items():
                    im = int(im_s)
                    for pr in lst:
                        obj = int(pr["obj_id"])
                        tag = tag_index.get(f"{scene}_{im}_{obj}")
                        if tag is None:
                            continue
                        fdir = ROOT / "bop6dof_results" / ds / tag
                        proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
                        if obj not in pts_cache:
                            pts_cache[obj] = eval_points_trimesh(BOP / ds / edir, BOP / ds / mdir, obj)
                            sym_cache[obj] = symmetry_transforms(info_all[str(obj)])
                        R_est = np.asarray(pr["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
                        t_est = np.asarray(pr["cam_t_m2c"], dtype=np.float64).reshape(3)
                        R_gt, t_gt = np.asarray(proto["gt_R"], dtype=np.float64), np.asarray(proto["gt_t"], dtype=np.float64)
                        K = np.load(fdir / "observation.npz")["K"]
                        met = compute_metrics(R_est, t_est, R_gt, t_gt, K, pts_cache[obj], info_all[str(obj)], sym_cache[obj])
                        init = next((i for i in proto["inits"] if i["bin"] == bname), None)
                        recs.append({"tag": tag, "dataset": ds, "scene_id": scene, "im_id": im, "obj_id": obj, "method": "scflow2_refine",
                                     "bin": bname, "init_rot_err_deg": (init or {}).get("rot_err_deg"),
                                     "R_est": R_est.tolist(), "t_est": t_est.tolist(), "t_total_s": float(pr.get("time", float("nan"))), **met})
        if recs:
            out.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
            e = np.array([r["e_R_deg"] for r in recs])
            print(f"[{ds}] {len(recs)} records -> {out.name}; SR@10 raw {100*np.mean(e<=10):.1f}%  ADD {100*np.mean([r['add_lt_0.1d'] for r in recs]):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
