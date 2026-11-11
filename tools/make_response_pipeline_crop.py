#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure R3 of the response letter: the pipeline figure (Figure 1 of the manuscript) cropped from the compiled
manuscript as vector graphics, so that the letter always shows the figure of the current PDF.

  python tools/make_response_pipeline_crop.py [--pdf manuscript/computational_visual_media/main_pr.pdf]

Writes figures/response/fig_r_pipeline.pdf next to the manuscript. Requires PyMuPDF.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manuscript" / "computational_visual_media"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(MAN / "main_pr.pdf"))
    ap.add_argument("--out", default="")
    ap.add_argument("--margin", type=float, default=4.0, help="margin around the figure in pt")
    args = ap.parse_args()
    src = fitz.open(args.pdf)
    page = next((p for p in src if "Figure 1:" in p.get_text()), None)
    if page is None:
        print("Figure 1 not found")
        return 1
    cap_top = min(b[1] for b in page.get_text("blocks") if b[4].lstrip().startswith("Figure 1:"))
    boxes = [d["rect"] for d in page.get_drawings() if d["rect"].y1 < cap_top and d["rect"].height > 0.5]
    boxes += [r for im in page.get_images(full=True) for r in page.get_image_rects(im[0]) if r.y1 < cap_top]
    clip = fitz.Rect(min(r.x0 for r in boxes) - args.margin, min(r.y0 for r in boxes) - args.margin,
                     max(r.x1 for r in boxes) + args.margin, max(r.y1 for r in boxes) + args.margin)
    out = Path(args.out) if args.out else Path(args.pdf).resolve().parent / "figures" / "response" / "fig_r_pipeline.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    dst = fitz.open()
    dst.new_page(width=clip.width, height=clip.height).show_pdf_page(fitz.Rect(0, 0, clip.width, clip.height), src, page.number, clip=clip)
    dst.save(str(out), garbage=4, deflate=True)
    print(f"page {page.number + 1}, clip {[round(v, 1) for v in clip]} -> {out} ({clip.width:.0f} x {clip.height:.0f} pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
