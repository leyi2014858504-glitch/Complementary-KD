# -*- coding: utf-8 -*-
"""Audit: every citation key used in styled tex exists in cas-refs.bib."""
import re

tex = open(r"d:\project ML\Complementary_KD_styled (1).tex", encoding="utf-8").read()
bib = open(r"d:\project ML\cas-refs.bib", encoding="utf-8").read()

used = set()
for m in re.finditer(r"\\cite[tp]?\{([^}]*)\}", tex):
    for k in m.group(1).split(","):
        k = k.strip()
        if k:
            used.add(k)
defined = set(re.findall(r"@\w+\{([^,]+),", bib))

print("used but NOT defined:", sorted(used - defined))
print("defined but unused:", sorted(defined - used))
