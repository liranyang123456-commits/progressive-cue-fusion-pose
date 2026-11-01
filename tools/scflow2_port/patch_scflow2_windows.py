"""Source patches that make the public SCFlow2 code base (W-QY/SCFlow2) run on Windows.

Usage:  python patch_scflow2_windows.py <SCFlow2 checkout>  [--data-root <SCFlow2>/data]

All patches are path handling only; no numerical code is touched:
  1. image lists are joined to data_root with forward slashes;
  2. ``osp.join(root, "{:06}/...")`` templates are built by concatenation
     (``ntpath.join`` treats ``{:`` as a drive specifier and drops the root);
  3. ``os.path`` is replaced by ``posixpath`` in datasets/, models/ and test.py so that all
     glob/join calls produce forward-slash paths on Windows (the annotation keys depend on it).
The ``image_lists/ours.txt`` files written by tools/make_scflow2_data.py are made relative to data_root.
"""
from __future__ import annotations

import argparse
import pathlib
import re


def patch_base_dataset(root: pathlib.Path) -> None:
    p = root / "datasets" / "base_dataset.py"
    s = p.read_text(encoding="utf-8")
    old = "img_files = [osp.join(self.data_root, x.strip()) for x in img_files]"
    new = ("img_files = [self.data_root.rstrip('/') + '/' + x.strip().replace('\\\\', '/') for x in img_files]"
           "  # forward slashes on every OS")
    if old in s:
        p.write_text(s.replace(old, new, 1), encoding="utf-8")
        print("patched datasets/base_dataset.py (image list join)")
    else:
        print("base_dataset.py:", "already patched" if new in s else "ANCHOR NOT FOUND")


def patch_join_templates(root: pathlib.Path) -> None:
    n = 0
    for rel in ("datasets/refine.py", "datasets/base_dataset.py", "datasets/estimate.py", "datasets/supervise_refine.py"):
        p = root / rel
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8")
        s2, k = re.subn(r'osp\.join\((self\.\w+),\s*"(\{:06d?\}/[^"]+)"\)', r'(\1.rstrip("/") + "/" + "\2")', s)
        if k:
            p.write_text(s2, encoding="utf-8")
            print(f"patched {rel} ({k} template joins)")
            n += k
    print("template joins patched:", n)


def patch_posixpath(root: pathlib.Path) -> None:
    n_mod = 0
    files = list((root / "datasets").rglob("*.py")) + list((root / "models").rglob("*.py")) + [root / "test.py"]
    for p in files:
        s = p.read_text(encoding="utf-8")
        s2 = re.sub(r"^(\s*)from os import path as osp\s*$", r"\1import posixpath as osp  # forward-slash joins on Windows", s, flags=re.M)
        s2 = re.sub(r"^(\s*)import os\.path as osp\s*$", r"\1import posixpath as osp  # forward-slash joins on Windows", s2, flags=re.M)
        s2 = s2.replace("int(p.split('/')[-1][4:10])", "int(p.replace('\\\\', '/').split('/')[-1][4:10])")
        if s2 != s:
            p.write_text(s2, encoding="utf-8")
            n_mod += 1
            print("patched", p.relative_to(root).as_posix())
    print("files switched to posixpath:", n_mod)


def relativise_image_lists(data_root: pathlib.Path) -> None:
    for ds in ("lmo", "lm", "tless"):
        q = data_root / ds / "image_lists" / "ours.txt"
        if not q.exists():
            continue
        lines = [ln.strip() for ln in q.read_text(encoding="utf-8").splitlines() if ln.strip()]
        lines = [ln.replace(f"data/{ds}/test/", "") for ln in lines]
        q.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(ds, len(lines), "images;", "first:", lines[0] if lines else "-")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="SCFlow2 checkout")
    ap.add_argument("--data-root", default="", help="defaults to <root>/data")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    data_root = pathlib.Path(args.data_root) if args.data_root else root / "data"
    patch_base_dataset(root)
    patch_join_templates(root)
    patch_posixpath(root)
    relativise_image_lists(data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
