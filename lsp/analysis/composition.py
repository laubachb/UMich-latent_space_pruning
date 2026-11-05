"""Category composition of the selected training sets vs retention.

For every selection method and retention level, the frames of the ten replicate
cuts are pooled and counted per structure category; the table holds the share (%)
of each category in the pooled cut, plus the full dataset's shares (100 %).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import data, paths, selection


def run(dataset: str, methods: list[str] | None = None) -> pd.DataFrame:
    frames = data.load_frames(dataset)
    names, cat = data.categories(frames)
    methods = methods or selection.available(dataset)
    rows = []
    for m in methods:
        sel = selection.load(dataset, m)
        counts = {p: np.zeros(len(names)) for p in sel["percents"]}
        for p, _r, idx in selection.iter_cuts(sel):
            counts[p] += np.bincount(cat[idx], minlength=len(names))
        for p, c in counts.items():
            rows.append(dict(method=m, percent=p, n_pooled=int(c.sum()),
                             **{n: 100 * c[i] / c.sum() for i, n in enumerate(names)}))
        full = np.bincount(cat, minlength=len(names))
        rows.append(dict(method=m, percent=100, n_pooled=int(full.sum()),
                         **{n: 100 * full[i] / full.sum() for i, n in enumerate(names)}))
    df = pd.DataFrame(rows)
    out = paths.results_dir(dataset, "analysis") / "composition.csv"
    df.to_csv(out, index=False)
    print(f"[{dataset}] composition of {len(methods)} methods -> {out}")
    return df
