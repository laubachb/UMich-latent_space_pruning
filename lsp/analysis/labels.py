"""Label-space coverage: how well does each cut reproduce the distribution of the DFT labels?

For every cut S and three labels of the full training set,

    energy per atom      one value per frame (eV/atom)
    force magnitude      one value per atom, |F_i| (eV/A)
    mean normal stress   one value per frame, (s_xx + s_yy + s_zz) / 3 of mlearn's virial_stress

the distance between the label distribution of S and that of the full set is the
1-Wasserstein (earth mover's) distance divided by the standard deviation of the full
distribution, W1(S, full) / sigma_full. It is dimensionless (independent of the stress
unit), 0 when S reproduces the full distribution, and comparable across labels and
elements; for orientation, a cut that put all its mass at the full mean would score
about 0.8 for a Gaussian label.

If every latent space covers these distributions about equally well, models trained
on their cuts should see the same range of energies, forces and stresses — the most
direct mechanism for their predictions converging.

Output: results/<ds>/analysis/label_coverage.csv (one row per cut)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .. import data, paths, selection

LABELS = {"energy_per_atom": "Energy per atom", "force_magnitude": "Force magnitude",
          "mean_normal_stress": "Mean normal stress"}


def label_arrays(dataset: str) -> dict:
    """Per-frame label values; force magnitudes as one array per frame."""
    frames = data.load_frames(dataset)
    e, s, f = [], [], []
    for d in frames:
        out = d["outputs"]
        n = len(d["structure"])
        e.append(out["energy"] / n)
        vs = np.asarray(out["virial_stress"], dtype=float)
        s.append(vs[:3].mean())
        f.append(np.linalg.norm(np.asarray(out["forces"], dtype=float), axis=1))
    return dict(energy_per_atom=np.array(e), mean_normal_stress=np.array(s), force_magnitude=f)


def run(dataset: str, methods: list[str] | None = None) -> pd.DataFrame:
    L = label_arrays(dataset)
    full = {"energy_per_atom": L["energy_per_atom"], "mean_normal_stress": L["mean_normal_stress"],
            "force_magnitude": np.concatenate(L["force_magnitude"])}
    sigma = {k: float(np.std(v)) for k, v in full.items()}
    methods = methods or selection.available(dataset)
    t0 = time.time()
    rows = []
    for m in methods:
        sel = selection.load(dataset, m)
        for p, r, idx in selection.iter_cuts(sel):
            row = dict(method=m, percent=p, replicate=r + 1, n_frames=len(idx))
            sub = {"energy_per_atom": L["energy_per_atom"][idx], "mean_normal_stress": L["mean_normal_stress"][idx],
                   "force_magnitude": np.concatenate([L["force_magnitude"][i] for i in idx])}
            for k in LABELS:
                row[f"w1_{k}"] = wasserstein_distance(sub[k], full[k]) / sigma[k]
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["method", "percent", "replicate"])
    for k in LABELS:
        df[f"sigma_full_{k}"] = sigma[k]
    out = paths.results_dir(dataset, "analysis") / "label_coverage.csv"
    df.to_csv(out, index=False)
    low = df[df.percent <= 10].groupby("method")[[f"w1_{k}" for k in LABELS]].mean()
    print(f"[{dataset}] label coverage: {len(df)} cuts in {time.time() - t0:.0f} s; "
          f"W1/sigma at <=10 % retention, random: "
          + ", ".join(f"{k} {low.loc['random', f'w1_{k}']:.2f}" for k in LABELS if "random" in low.index))
    return df
