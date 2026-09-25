# -*- coding: utf-8 -*-
"""Inspect Response_Letter_Revised.docx: locate HASKD, §3.1, R1C5 paragraphs."""
from docx import Document

d = Document(r"d:\project ML\Response_Letter_Revised.docx")
for i, p in enumerate(d.paragraphs):
    t = p.text
    if any(k in t for k in ("HASKD", "\u00a73.1", "swept grid", "Comment 5",
                            "label-smoothing comparison", "Dataset)")):
        print(f"[{i}] runs={len(p.runs)}")
        print("   ", t[:600].replace("\n", " | "))
        for j, r in enumerate(p.runs):
            if "HASKD" in r.text or "\u00a73.1" in r.text or "swept grid" in r.text:
                print(f"    run{j}: {r.text[:200]!r}")
        print()
