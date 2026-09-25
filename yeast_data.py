"""Yeast loader via direct UCI download (OpenML API is unreliable on this network).

Dataset: Yeast, UCI ML Repository (doi 10.24432/C5VG87)
  1484 samples, 8 numeric attributes, 10 classes (location sites).
  Class sizes are highly imbalanced (min class = 5 samples), which also
  stress-tests the teacher class-alignment fix in tabular_ext.

Saves a cached npz so later runs skip the download.
"""
import os
import urllib.request

import numpy as np

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'
CACHE = os.path.join('data', 'yeast.npz')


def load_yeast(cache=CACHE):
    if os.path.exists(cache):
        z = np.load(cache)
        return z['X'], z['y']

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    print(f'Downloading {URL} ...', flush=True)
    raw = urllib.request.urlopen(URL, timeout=60).read().decode('ascii')
    rows = [ln.split() for ln in raw.strip().splitlines() if ln.strip()]
    feats = np.array([[float(v) for v in r[1:9]] for r in rows], dtype=np.float64)
    labels = [r[9] for r in rows]
    classes = sorted(set(labels))
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[l] for l in labels], dtype=np.int64)
    np.savez(cache, X=feats, y=y, classes=np.array(classes))
    print(f'Loaded yeast: X={feats.shape}, classes={classes}', flush=True)
    return feats, y


if __name__ == '__main__':
    X, y = load_yeast()
    classes, counts = np.unique(y, return_counts=True)
    print('class sizes:', dict(zip(classes.tolist(), counts.tolist())))
    print('min class size:', counts.min(), '| any NaN:', bool(np.isnan(X).any()))
