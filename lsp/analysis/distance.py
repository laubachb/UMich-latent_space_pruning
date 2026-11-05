"""Category-resolved latent-space distances and their relation to selection.

For every latent space: standardise, take all pairwise euclidean distances and
divide by the space's maximum (support [0, 1]). Each directed distance i -> j is
owned by the category of its source frame i, so the per-category histograms stack
exactly to the full distribution. For each category we report its mean normalised
distance to the other frames, its within-category mean distance, and — for every
sampler — its enrichment at low retention (share in the pooled cuts at retention
<= 10 % divided by its share of the full dataset); Spearman rho(mean distance,
enrichment) across categories summarises how strongly isolation in the latent
space drives selection by that sampler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from .. import data, descriptors, paths, selection

BINS = 50
LOW_RETENTION_MAX = 10


def normalized_distance_matrix(X: np.ndarray) -> np.ndarray:
    ok = np.isfinite(X).all(axis=1)
    d = np.full((len(X), len(X)), np.nan)
    sub = squareform(pdist(descriptors.standardize(X[ok])))
    sub /= sub.max()
    np.fill_diagonal(sub, np.nan)
    d[np.ix_(ok, ok)] = sub
    return d


def run(dataset: str, names: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = data.load_frames(dataset)
    cats, cat = data.categories(frames)
    names = names or descriptors.available(dataset)
    comp = pd.read_csv(paths.results_dir(dataset, "analysis") / "composition.csv")
    full_share = np.bincount(cat, minlength=len(cats)) / len(cat)
    edges = np.linspace(0, 1, BINS + 1)
    hist_rows, sel_rows = [], []
    for name in names:
        d = normalized_distance_matrix(descriptors.compute(dataset, name))
        n_ok = int(np.isfinite(d).any(axis=1).sum())
        for c, cname in enumerate(cats):
            vals = d[cat == c]
            vals = vals[np.isfinite(vals)]
            dens = np.histogram(vals, bins=edges)[0] / (n_ok * (n_ok - 1) * (edges[1] - edges[0]))
            for b in range(BINS):
                hist_rows.append(dict(descriptor=name, category=cname, bin_left=edges[b], density=dens[b]))
        mean_d = np.array([np.nanmean(d[cat == c]) for c in range(len(cats))])
        intra = np.array([np.nanmean(d[np.ix_(cat == c, cat == c)]) if (cat == c).sum() > 1 else np.nan
                          for c in range(len(cats))])
        for sampler in selection.SAMPLERS:
            m = selection.method_id(sampler, name)
            low = comp[(comp.method == m) & (comp.percent <= LOW_RETENTION_MAX)]
            if not len(low):
                continue
            enrich = low[cats].mean(axis=0).to_numpy() / 100 / full_share
            ok = np.isfinite(mean_d) & np.isfinite(enrich)
            rho = spearmanr(mean_d[ok], enrich[ok])[0] if ok.sum() > 2 else np.nan
            for c, cname in enumerate(cats):
                sel_rows.append(dict(descriptor=name, sampler=sampler, category=cname, category_id=c,
                                     mean_distance=mean_d[c], within_distance=intra[c],
                                     enrichment=enrich[c], spearman_rho=rho))
            print(f"[{dataset}] {name} / {sampler}: rho(mean distance, enrichment <= {LOW_RETENTION_MAX}%) = {rho:.2f}")
    hist = pd.DataFrame(hist_rows)
    sel = pd.DataFrame(sel_rows)
    out = paths.results_dir(dataset, "analysis")
    hist.to_csv(out / "distance_histograms.csv", index=False)
    sel.to_csv(out / "distance_selection.csv", index=False)
    return hist, sel
