#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pack the exported BOP frame packages + the needed object models for a GPU worker (bop_full6dof_eval.py --from-export)."""
from __future__ import annotations
import os

import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOP = Path(os.environ.get("BOP_ROOT", "data/bop"))
CFG = {"lm": ("models", "models_eval"), "lmo": ("models", "models_eval"), "tless": ("models_cad", "models_eval")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", default="")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for c in ("bop_full6dof_eval.py", "bop_metrics.py"):
        shutil.copy2(ROOT / c, out / c)
    objs: dict[str, set[int]] = {}
    n = 0
    for ds in CFG:
        src = ROOT / "bop6dof_results" / ds
        if not src.exists():
            continue
        for d in sorted(src.iterdir()):
            if not (d / "protocol.json").exists():
                continue
            pr = json.loads((d / "protocol.json").read_text(encoding="utf-8"))
            objs.setdefault(ds, set()).add(int(pr["obj_id"]))
            dst = out / "bop6dof_results" / ds / d.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("observation.npz", "protocol.json"):
                shutil.copy2(d / f, dst / f)
            n += 1
    for ds, (mdir, edir) in CFG.items():
        for sub in (mdir, edir):
            s = BOP / ds / sub
            t = out / "bop_data" / ds / sub
            t.mkdir(parents=True, exist_ok=True)
            if (s / "models_info.json").exists():
                shutil.copy2(s / "models_info.json", t / "models_info.json")
            for o in sorted(objs.get(ds, ())):
                f = s / f"obj_{o:06d}.ply"
                if f.exists():
                    shutil.copy2(f, t / f.name)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"[bop-bundle] {n} frames, objects={ {k: sorted(v) for k, v in objs.items()} }, {total/1e6:.0f} MB -> {out}")
    if args.zip:
        zp = Path(args.zip)
        with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in out.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(out).as_posix())
        print(f"[bop-bundle] zip {zp} ({zp.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
