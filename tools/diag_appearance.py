#!/usr/bin/env python3
"""
Diagnostic for the flipped-basin failures on LM-O / LM: does an appearance score built on foundation features
(DINOv2 patch tokens of a vertex-colour render against the observed RGB crop) prefer the correct basin over the
proposed method's estimate, and do 180-degree flips of that estimate cover the correct basin?
Ground truth is used only to label the candidates.

  C:/Users/lry/.conda/envs/kaolin/python.exe tools/diag_appearance.py
Writes benchmark_results/diag_appearance.json.
"""
from __future__ import annotations

import os

import importlib.util
import json
from pathlib import Path

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import trimesh

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp_bop", Path(__file__).resolve().parent / "export_bop_results.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)
eb.RES = ROOT / "bop6dof_results"
eb.BOP_ROOT = Path(os.environ.get("BOP_ROOT", "data/bop"))
DINO_REPO = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CROP = 224


def rot_axis(a: np.ndarray, ang: float) -> np.ndarray:
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


class Renderer:
    def __init__(self):
        self.glctx = dr.RasterizeCudaContext()
        self.cache = {}

    def mesh(self, ds: str, obj: int):
        if (ds, obj) not in self.cache:
            m = trimesh.load(str(eb.BOP_ROOT / ds / "models" / f"obj_{obj:06d}.ply"), process=False)
            v = np.asarray(m.vertices, dtype=np.float32)
            col = np.asarray(m.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
            centre = v.mean(axis=0)
            _, _, Vt = np.linalg.svd(v - centre, full_matrices=False)
            self.cache[(ds, obj)] = (torch.from_numpy(v).cuda(), torch.from_numpy(np.asarray(m.faces, dtype=np.int32)).cuda(),
                                     torch.from_numpy(col).cuda(), centre.astype(np.float64), Vt.astype(np.float64))
        return self.cache[(ds, obj)]

    def render(self, ds, obj, R, t, K, H, W):
        v, f, col, _, _ = self.mesh(ds, obj)
        Rt = torch.as_tensor(R, dtype=torch.float32, device="cuda")
        tt = torch.as_tensor(t, dtype=torch.float32, device="cuda")
        pc = v @ Rt.T + tt
        X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        n, fa = 10.0, 20000.0
        pos = torch.stack([(2 * fx * X + (2 * cx + 1 - W) * Z) / W, (2 * fy * Y + (2 * cy + 1 - H) * Z) / H,
                           (Z * (fa + n) - 2 * fa * n) / (fa - n), Z], dim=-1)[None]
        rast, _ = dr.rasterize(self.glctx, pos, f, resolution=[H, W])
        rgb, _ = dr.interpolate(col[None], rast, f)
        return rgb[0].clamp(0, 1), rast[0, ..., 3] > 0


def crop_box(mask: np.ndarray, scale: float = 1.2):
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    side = scale * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    return cx - side / 2, cy - side / 2, side


def crop(img: torch.Tensor, box, H, W) -> torch.Tensor:
    """img (C,H,W) -> (C,CROP,CROP) via an affine grid (zero padding outside the image)."""
    x0, y0, side = box
    lin = (torch.arange(CROP, device=img.device, dtype=torch.float32) + 0.5) / CROP * side
    gx = (x0 + lin) / W * 2 - 1
    gy = (y0 + lin) / H * 2 - 1
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], dim=-1)[None]
    return F.grid_sample(img[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]


class Dino:
    def __init__(self):
        self.m = torch.hub.load(str(DINO_REPO), "dinov2_vitl14", source="local").cuda().eval()

    @torch.no_grad()
    def __call__(self, img: torch.Tensor):
        x = ((img.cpu() - MEAN) / STD).cuda()[None]
        out = self.m.forward_features(x)
        return F.normalize(out["x_norm_patchtokens"][0], dim=-1), F.normalize(out["x_norm_clstoken"][0], dim=-1)


def main() -> int:
    ren, dino = Renderer(), Dino()
    files = {"ours": "results.jsonl", "wapr12": "baseline_wapr_full.jsonl", "pi3ours": "results_init_pi3.jsonl"}
    out, summary = [], {}
    for ds in ("lmo", "lm"):
        recs = {}
        for k, fn in files.items():
            rows = [r for r in eb.read_jsonl(eb.RES / ds / fn) if "R_est" in r and r.get("R_est") is not None]
            eb.add_sym_error(ds, rows)
            recs[k] = {r["tag"]: r for r in rows}
        for tag, ro in sorted(recs["ours"].items()):
            fdir = eb.RES / ds / tag
            proto = json.loads((fdir / "protocol.json").read_text(encoding="utf-8"))
            obs = np.load(fdir / "observation.npz")
            obj = int(proto["obj_id"])
            K = np.asarray(obs["K"], dtype=np.float64)
            rgb = torch.from_numpy(obs["rgb"].astype(np.float32) / 255.0).permute(2, 0, 1).cuda()
            m_obs = obs["mask"] > 0
            H, W = m_obs.shape
            Rg, tg = np.array(proto["gt_R"]), np.array(proto["gt_t"])
            syms = eb.sym_rots(ds, obj)
            esym = lambda R: float(min(eb.rotation_error_deg(R, Rg @ S) for S in syms))
            _, _, _, centre, Vt = ren.mesh(ds, obj)
            Ro, to = np.array(ro["R_est"]), np.array(ro["t_est"])
            cands = {"gt": (Rg, tg), "ours": (Ro, to)}
            for k in ("wapr12", "pi3ours"):
                if tag in recs[k]:
                    cands[k] = (np.array(recs[k][tag]["R_est"]), np.array(recs[k][tag]["t_est"]))
            pc = Ro @ centre + to
            for i in range(3):  # 180 deg about the object's principal axes (object frame), centre kept in place
                Rf = Ro @ rot_axis(Vt[i], np.pi)
                cands[f"flip_obj{i}"] = (Rf, pc - Rf @ centre)
            for i, ax in enumerate((np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))):  # 180 deg about camera x / y through the centre
                Rf = rot_axis(ax, np.pi) @ Ro
                cands[f"flip_cam{'xy'[i]}"] = (Rf, pc - Rf @ centre)
            box = crop_box(m_obs)
            o_img = crop(rgb * torch.from_numpy(m_obs).cuda().float()[None], box, H, W)
            o_msk = crop(torch.from_numpy(m_obs).cuda().float()[None], box, H, W)[0]
            pm = F.avg_pool2d(o_msk[None, None], 14)[0, 0].flatten() >= 0.5
            fo, co = dino(o_img)
            row = {"dataset": ds, "tag": tag, "obj_id": obj, "cands": {}}
            for name, (R, t) in cands.items():
                r_img, r_msk = ren.render(ds, obj, R, t, K, H, W)
                fr_, cr_ = dino(crop(r_img.permute(2, 0, 1), box, H, W))
                patch = float((fo[pm] * fr_[pm]).sum(-1).mean()) if pm.any() else 0.0
                iou = float((r_msk & torch.from_numpy(m_obs).cuda()).sum() / max(1, (r_msk | torch.from_numpy(m_obs).cuda()).sum()))
                row["cands"][name] = {"e_R_sym": esym(R), "patch": patch, "cls": float((co * cr_).sum()), "iou": iou}
            out.append(row)
            c = row["cands"]
            print(f"[{ds} {tag}] ours {c['ours']['e_R_sym']:6.1f}deg patch {c['ours']['patch']:.3f} | gt patch {c['gt']['patch']:.3f} | "
                  + " ".join(f"{k}:{v['e_R_sym']:.0f}/{v['patch']:.3f}" for k, v in c.items() if k.startswith("flip")), flush=True)

    for ds in ("lmo", "lm", "all"):
        rows = [r for r in out if ds == "all" or r["dataset"] == ds]
        fail = [r for r in rows if r["cands"]["ours"]["e_R_sym"] > 10]
        okr = [r for r in rows if r["cands"]["ours"]["e_R_sym"] <= 10]
        flipk = lambda r: [k for k in r["cands"] if k.startswith("flip")]
        s = {
            "n": len(rows), "n_fail": len(fail),
            "gt_beats_ours_patch": sum(r["cands"]["gt"]["patch"] > r["cands"]["ours"]["patch"] for r in fail),
            "gt_beats_ours_cls": sum(r["cands"]["gt"]["cls"] > r["cands"]["ours"]["cls"] for r in fail),
            "flip_covers_basin30": sum(any(r["cands"][k]["e_R_sym"] <= 30 for k in flipk(r)) for r in fail),
            "select_patch_basin30_fail": sum(r["cands"][max(["ours"] + flipk(r), key=lambda k: r["cands"][k]["patch"])]["e_R_sym"] <= 30 for r in fail),
            "select_patch_keeps_ok": sum(max(["ours"] + flipk(r), key=lambda k: r["cands"][k]["patch"]) == "ours" for r in okr),
            "n_ok": len(okr),
        }
        summary[ds] = s
        print(ds, s)
    (ROOT / "benchmark_results").mkdir(exist_ok=True)
    (ROOT / "benchmark_results" / "diag_appearance.json").write_text(json.dumps({"summary": summary, "frames": out}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
