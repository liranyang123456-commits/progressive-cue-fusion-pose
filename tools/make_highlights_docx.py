#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write highlights.txt / highlights.docx (five bullets, <= 85 characters each) for the revised submission."""
from __future__ import annotations

from pathlib import Path

MAN = Path(__file__).resolve().parents[1] / "manuscript" / "computational_visual_media"
TITLE = "Known Geometry Camera Pose Refinement via Progressive Cue Fusion"
BULLETS = [
    "Progressive cue fusion: geometry proposes, depth sets the shell, RGB only selects.",
    "Metric depth hint and multi-seed search recover far-off camera poses, no training.",
    "Theory bounds spurious minima; controls show the retained seeds protect the basin.",
    "Sub-degree median error on five meshes, more reliable than WAPR on the same views.",
    "As a front end it lifts six real-data refiners, GoTrack and MegaPose included.",
]


def main() -> int:
    for b in BULLETS:
        assert len(b) <= 85, (len(b), b)
    txt = ["Highlights", TITLE, "Pattern Recognition (Ms. Ref. No.: PR-D-26-09325)", ""] + [f"- {b}" for b in BULLETS] + ["", "Character counts: " + " / ".join(str(len(b)) for b in BULLETS) + " (each at most 85)."]
    (MAN / "highlights.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    try:
        from docx import Document
        from docx.shared import Pt
    except Exception:
        print("python-docx not available; highlights.txt written only")
        return 0
    doc = Document()
    doc.add_heading("Highlights", level=1)
    p = doc.add_paragraph(TITLE)
    p.runs[0].italic = True
    for b in BULLETS:
        para = doc.add_paragraph(style="List Bullet")
        run = para.add_run(b)
        run.font.size = Pt(11)
    doc.save(MAN / "highlights.docx")
    print("highlights written:", [len(b) for b in BULLETS])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
