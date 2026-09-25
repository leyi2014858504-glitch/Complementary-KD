# -*- coding: utf-8 -*-
"""Edit Response_Letter_Revised.docx in place:
1) HASKD -> HSAKD (all runs)
2) §3.1 -> §4.1 (Datasets change location)
3) Comment 5 response: append seven-dataset LS-table sentence
4) Comment 5 change location: add §4.8
Prints full text of touched paragraphs for verification.
"""
from docx import Document

P = r"d:\project ML\Response_Letter_Revised.docx"
d = Document(P)

def replace_in_runs(para, old, new):
    hit = False
    for r in para.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            hit = True
    return hit

# 1) HASKD -> HSAKD everywhere
n1 = 0
for p in d.paragraphs:
    if replace_in_runs(p, "HASKD", "HSAKD"):
        n1 += 1

# 2) §3.1 -> §4.1
n2 = 0
for p in d.paragraphs:
    if replace_in_runs(p, "\u00a73.1", "\u00a74.1"):
        n2 += 1

# 3) Comment 5 response append
resp = d.paragraphs[16]
assert resp.text.startswith("Response."), f"unexpected [16]: {resp.text[:60]}"
print("--- [16] BEFORE (full) ---")
print(resp.text)
add = (" The \u03b1 = 0 comparison with label smoothing now covers all seven "
       "datasets: Distill-S beats LS in all 20 seeds on Yeast, Fashion-MNIST "
       "and CIFAR-10 (Holm-adjusted p < 0.001), and on Yeast and CIFAR-10 "
       "label smoothing even increases the gap relative to the hard-label "
       "baseline.")
if "seven datasets" not in resp.text:
    resp.runs[-1].text = resp.runs[-1].text.rstrip() + add

# 4) Comment 5 change location append
loc = d.paragraphs[17]
assert loc.text.startswith("Change location."), f"unexpected [17]: {loc.text[:60]}"
if "\u00a74.8" not in loc.text:
    body = loc.runs[-1]
    txt = body.text.rstrip()
    if txt.endswith("."):
        txt = txt[:-1]
    body.text = txt + "; \u00a74.8 (LS-comparison table, extended to seven datasets)."

d.save(P)
print("--- touched paragraphs AFTER ---")
for i in (4, 8, 16, 17, 41):
    print(f"[{i}]", d.paragraphs[i].text[:500])
print(f"\nHASKD paras fixed: {n1}; sec3.1 fixed: {n2}")
