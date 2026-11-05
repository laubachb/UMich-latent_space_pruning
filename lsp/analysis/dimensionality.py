"""Effective dimensionality of every latent space.

Participation ratio of the PCA eigenvalue spectrum of the standardised space,
PR = (sum_i l_i)^2 / sum_i l_i^2: the effective number of independent directions
the representation spreads the same frames over. Also reports the number of
components needed for 95 % of the variance and the summary of the normalised
distance distribution (mean / median / fraction of near-duplicate pairs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from .. import descriptors, paths

DUPLICATE_TOL = 1e-6


def run(dataset: str, names: list[str] | None = None) -> pd.DataFrame:
    names = names or descriptors.available(dataset)
    rows = []
    for name in names:
        X = descriptors.compute(dataset, name)
        ok = np.isfinite(X).all(axis=1)
        Z = descriptors.standardize(X[ok])
        lam = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2 / (len(Z) - 1)
        pr = lam.sum() ** 2 / (lam ** 2).sum()
        n95 = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1)
        d = pdist(Z)
        dn = d / d.max()
        rows.append(dict(descriptor=name, n_frames=int(ok.sum()), n_features=X.shape[1],
                         participation_ratio=pr, n_components_95pct=n95,
                         mean_norm_distance=dn.mean(), median_norm_distance=np.median(dn),
                         duplicate_pairs=int((d < DUPLICATE_TOL * d.max()).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(paths.results_dir(dataset, "analysis") / "dimensionality.csv", index=False)
    print(df.to_string(index=False))
    return df
