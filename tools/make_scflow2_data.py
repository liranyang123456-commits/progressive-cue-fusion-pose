#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the input tree that SCFlow2 (CVPR 2025, github.com/W-QY/SCFlow2) expects for OUR protocol frames,
so that SCFlow2 can be run as an additional refiner from exactly the same initial poses as WAPR / Diff-DOPE /
our stages II-IV.

For every exported frame package (bop6dof_results/<ds>/<tag>/{observation.npz, protocol.json}) the script
writes a minimal BOP-layout test set (rgb, depth, visible mask, scene_camera/gt/gt_info), one "reference
pose" tree per initial-pose bin (SCFlow2 reads the initial pose from <ref_annots_root>/<scene>/scene_gt.json),
the mesh folders in the naming SCFlow2 uses (LM: 13 contiguous objects; T-LESS: 30), the bounding-box keypoint
files, image lists and the mmcv config files.

  python tools/make_scflow2_data.py --out external/scflow2_data

The tree is then copied to the GPU worker and run with
  python test.py --config configs/ours/<ds>_<bin>.py --checkpoint checkpoints/scflow2_pretrained.pth --save-dir results/ours/<ds>_<bin>
"""
from __future__ import annotations

import os

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bop6dof_results"
BOP = Path(os.environ.get("BOP_ROOT", "data/bop"))
BINS = ["15-30", "30-60", "60-90", "90-180"]
LM_13 = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]  # SCFlow2's LM convention: bowl (3) and cup (7) removed, renumbered 1..13
LM_MAP = {o: i + 1 for i, o in enumerate(LM_13)}
TLESS_OBJS = list(range(1, 31))


def bbox_from_mask(m: np.ndarray) -> list[int]:
    ys, xs = np.where(m)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def copy_models(src_dir: Path, dst_dir: Path, id_map: dict[int, int]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    info = json.loads((src_dir / "models_info.json").read_text(encoding="utf-8"))
    new_info = {}
    for src_id, dst_id in id_map.items():
        shutil.copy2(src_dir / f"obj_{src_id:06d}.ply", dst_dir / f"obj_{dst_id:06d}.ply")
        new_info[str(dst_id)] = info[str(src_id)]
    (dst_dir / "models_info.json").write_text(json.dumps(new_info, indent=1), encoding="utf-8")


def bbox_keypoints(mesh_dir: Path, out_json: Path) -> None:
    import trimesh
    pts = []
    for p in sorted(mesh_dir.glob("*.ply")):
        m = trimesh.load(str(p), force="mesh")
        pts.append(m.bounding_box.vertices.tolist())
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(pts, indent=2), encoding="utf-8")


def build_dataset(ds: str, out: Path) -> dict:
    data = out / "data"
    frames = sorted([d for d in (RES / ds).iterdir() if d.is_dir() and (d / "observation.npz").exists() and (d / "protocol.json").exists()])
    per_image: dict[tuple[int, int], list[tuple[Path, dict]]] = {}
    for fdir in frames:
        pr = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
        per_image.setdefault((int(pr["scene_id"]), int(pr["im_id"])), []).append((fdir, pr))
    test_root = data / ds / "test"
    cams: dict[int, dict] = {}
    gts: dict[int, dict] = {}
    infos: dict[int, dict] = {}
    refs: dict[str, dict[int, dict]] = {b: {} for b in BINS}
    image_list = []
    tag_index = {}  # (scene, im, obj) -> tag
    n_obj_dup = 0
    for (scene, im), lst in sorted(per_image.items()):
        seen_obj = set()
        sdir = test_root / f"{scene:06d}"
        for sub in ("rgb", "depth", "mask_visib"):
            (sdir / sub).mkdir(parents=True, exist_ok=True)
        gts.setdefault(scene, {}).setdefault(str(im), [])
        infos.setdefault(scene, {}).setdefault(str(im), [])
        for b in BINS:
            refs[b].setdefault(scene, {}).setdefault(str(im), [])
        wrote_image = False
        for fdir, pr in lst:
            obj = int(pr["obj_id"])
            if obj in seen_obj:  # SCFlow2 assumes one instance per object and image
                n_obj_dup += 1
                continue
            seen_obj.add(obj)
            z = np.load(fdir / "observation.npz")
            if not wrote_image:
                rgb = z["rgb"]
                cv2.imwrite(str(sdir / "rgb" / f"{im:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                depth_mm = z["depth"].astype(np.float64)
                cv2.imwrite(str(sdir / "depth" / f"{im:06d}.png"), np.clip(np.round(depth_mm * 10.0), 0, 65535).astype(np.uint16))
                K = np.asarray(z["K"], dtype=np.float64)
                cams.setdefault(scene, {})[str(im)] = {"cam_K": K.reshape(-1).tolist(), "depth_scale": 0.1}
                image_list.append(f"{scene:06d}/rgb/{im:06d}.png")  # relative to data_root (joined by the dataset class)
                wrote_image = True
            inst = len(gts[scene][str(im)])
            mask = z["mask"].astype(bool)
            cv2.imwrite(str(sdir / "mask_visib" / f"{im:06d}_{inst:06d}.png"), (mask * 255).astype(np.uint8))
            gts[scene][str(im)].append({"cam_R_m2c": np.asarray(pr["gt_R"], dtype=np.float64).reshape(-1).tolist(),
                                        "cam_t_m2c": np.asarray(pr["gt_t"], dtype=np.float64).reshape(-1).tolist(), "obj_id": obj})
            bb = bbox_from_mask(mask)
            npx = int(mask.sum())
            infos[scene][str(im)].append({"bbox_obj": bb, "bbox_visib": bb, "px_count_all": npx, "px_count_valid": npx,
                                          "px_count_visib": npx, "visib_fract": 1.0})
            for init in pr["inits"]:
                b = init["bin"]
                refs[b][scene][str(im)].append({"cam_R_m2c": np.asarray(init["R0"], dtype=np.float64).reshape(-1).tolist(),
                                                "cam_t_m2c": np.asarray(init["t0"], dtype=np.float64).reshape(-1).tolist(),
                                                "obj_id": obj, "score": 1.0})
            tag_index[f"{scene}_{im}_{obj}"] = fdir.name
    for scene in gts:
        sdir = test_root / f"{scene:06d}"
        (sdir / "scene_camera.json").write_text(json.dumps(cams[scene], indent=1), encoding="utf-8")
        (sdir / "scene_gt.json").write_text(json.dumps(gts[scene], indent=1), encoding="utf-8")
        (sdir / "scene_gt_info.json").write_text(json.dumps(infos[scene], indent=1), encoding="utf-8")
        for b in BINS:
            rdir = data / "initial_poses" / f"ours_bin{b}" / ds / f"{scene:06d}"
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / "scene_gt.json").write_text(json.dumps(refs[b][scene], indent=1), encoding="utf-8")
    (data / ds / "image_lists").mkdir(parents=True, exist_ok=True)
    (data / ds / "image_lists" / "ours.txt").write_text("\n".join(image_list) + "\n", encoding="utf-8")
    (data / ds / "tag_index.json").write_text(json.dumps(tag_index, indent=1), encoding="utf-8")
    objs = sorted({int(pr["obj_id"]) for lst in per_image.values() for _, pr in lst})
    print(f"[{ds}] {len(frames)} frames -> {len(image_list)} images, objects {objs}, duplicate-object instances skipped: {n_obj_dup}")
    return {"objects": objs, "n_images": len(image_list), "n_frames": len(frames)}


def write_configs(out: Path, ds_objs: dict[str, list[int]]) -> None:
    sc = ROOT / "external" / "SCFlow2"
    cfg_dir = out / "configs" / "ours"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    base_lmo = (sc / "configs" / "unseen_refine_datasets" / "unseen_lmo.py").read_text(encoding="utf-8")
    model_txt = (sc / "configs" / "flow_refine" / "scflow2.py").read_text(encoding="utf-8")
    # model section only: drop the base reference and the dataset_root / symmetry / diameter block of scflow2.py
    start = model_txt.index("model = dict(")
    model_only = "train_unseen=True\n\n" + model_txt[start:]
    for ds in ds_objs:
        if ds in ("lmo", "lm"):
            txt = base_lmo
            txt = txt.replace("data_root='data/lmo/test'", f"data_root='data/{ds}/test'")
            txt = txt.replace("image_list='data/lmo/image_lists/test_bop19.txt'", f"image_list='data/{ds}/image_lists/ours.txt'")
            txt = txt.replace("ref_annots_root = f'data/reference_poses/{init_pose_method}/lmo'", "ref_annots_root = 'data/initial_poses/ours_bin15-30/" + ds + "'")
            txt = txt.replace("OCCLINEMOD_LABELS = [1, 5, 6, 8, 9, 10, 11, 12]", f"OCCLINEMOD_LABELS = {ds_objs[ds]}")
        else:  # tless: 30 contiguous objects, no label mapping
            info = json.loads((BOP / "tless" / "models_cad" / "models_info.json").read_text(encoding="utf-8"))
            diam = [round(float(info[str(i)]["diameter"]), 3) for i in TLESS_OBJS]
            txt = base_lmo
            txt = txt.replace("dataset_root = 'data/lm'", "dataset_root = 'data/tless'")
            txt = txt.replace("data_root='data/lmo/test'", "data_root='data/tless/test'")
            txt = txt.replace("image_list='data/lmo/image_lists/test_bop19.txt'", "image_list='data/tless/image_lists/ours.txt'")
            txt = txt.replace("ref_annots_root = f'data/reference_poses/{init_pose_method}/lmo'", "ref_annots_root = 'data/initial_poses/ours_bin15-30/tless'")
            cls = "CLASS_NAMES = (" + ", ".join(f"'obj_{i:02d}'" for i in TLESS_OBJS) + ")"
            s0 = txt.index("CLASS_NAMES = (")
            s1 = txt.index("LABEL_MAPPING")
            txt = txt[:s0] + cls + "\n" + txt[s1:]
            txt = txt.replace("LABEL_MAPPING = {1:1, 2:2, 4:3, 5:4, 6:5, 8:6, 9:7, 10:8, 11:9, 12:10, 13:11, 14:12, 15:13}",
                              "LABEL_MAPPING = {i: i for i in range(1, 31)}")
            txt = txt.replace("OCCLINEMOD_LABELS = [1, 5, 6, 8, 9, 10, 11, 12]", f"OCCLINEMOD_LABELS = {ds_objs[ds]}")
            m0 = txt.index("mesh_diameter =[")
            m1 = txt.index("]", m0) + 1
            txt = txt[:m0] + f"mesh_diameter = {diam}" + txt[m1:]
            txt = txt.replace("symmetry_types = { # 1-base\n    'cls_8': {'x':180, 'y':180, 'z':180},\n    'cls_9': {'z':180},\n}", "symmetry_types = {}")
            txt = txt.replace("/models_eval_13obj", "/models_eval").replace("/models_13obj", "/models").replace("/keypoints/bbox_13obj.json", "/keypoints/bbox.json")
        # Windows: no dataloader workers
        txt = txt.replace("workers_per_gpu=2", "workers_per_gpu=0")
        (cfg_dir / f"ds_{ds}.py").write_text(txt, encoding="utf-8", newline="\n")
        (cfg_dir / f"scflow2_{ds}.py").write_text(f"_base_ = './ds_{ds}.py'\n\n" + model_only, encoding="utf-8", newline="\n")
        for b in BINS:
            key = b.replace("-", "to")
            (cfg_dir / f"{ds}_bin{key}.py").write_text(
                f"_base_ = './scflow2_{ds}.py'\n\n"
                f"data = dict(test=dict(ref_annots_root='data/initial_poses/ours_bin{b}/{ds}'))\n"
                f"work_dir = 'work_dirs/ours/{ds}_bin{key}'\n", encoding="utf-8", newline="\n")
    print(f"[configs] -> {cfg_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "external" / "scflow2_data"))
    ap.add_argument("--datasets", default="lmo,lm,tless")
    ap.add_argument("--res-root", default="bop6dof_results", help="results root holding the exported frame packages")
    args = ap.parse_args()
    global RES
    RES = ROOT / args.res_root
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "data"
    # meshes (SCFlow2 naming)
    copy_models(BOP / "lm" / "models", data / "lm" / "models_13obj", LM_MAP)
    copy_models(BOP / "lm" / "models_eval", data / "lm" / "models_eval_13obj", LM_MAP)
    bbox_keypoints(data / "lm" / "models_eval_13obj", data / "lm" / "keypoints" / "bbox_13obj.json")
    tl_map = {i: i for i in TLESS_OBJS}
    copy_models(BOP / "tless" / "models_reconst", data / "tless" / "models", tl_map)
    copy_models(BOP / "tless" / "models_eval", data / "tless" / "models_eval", tl_map)
    bbox_keypoints(data / "tless" / "models_eval", data / "tless" / "keypoints" / "bbox.json")
    ds_objs = {}
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        ds_objs[ds] = build_dataset(ds, out)["objects"]
    write_configs(out, ds_objs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
