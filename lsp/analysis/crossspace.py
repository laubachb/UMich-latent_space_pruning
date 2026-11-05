"""Cross-space comparison: how alike are the latent spaces, and how alike are the cuts they select?

Geometry (per aggregation variant, on the frames defined in every space):
    rank agreement     Spearman correlation between the pairwise frame distances of two
                       standardised spaces (global layout: do they agree which frames are far apart?)
    kNN overlap        mean over frames of the Jaccard index of the k nearest neighbours of a
                       frame in the two spaces (local layout: do they agree on each frame's neighbours?)
                       Reference: two random k-sets of n-1 frames overlap by about k / (2(n-1) - k).

Selection (per sampler and variant, at every retention):
    between spaces     Jaccard index of the cuts of two different spaces with different replicate
                       seeds (r and r+1), averaged over replicates; mean, min and max over the
                       space pairs. (Same-seed pairs would share FPS's random first frame.)
    same space         Jaccard index of two replicates (seeds r, r+1) of the same space,
                       averaged over spaces: the seed-to-seed variability of the sampler itself
    random             the same for two independent random cuts (chance overlap)

Outputs (results/<ds>/analysis/): crossspace_geometry.csv, cut_overlap.csv
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from .. import config, descriptors, paths, selection

K_NEIGHBOURS = 10
VARIANTS = {"": "mean_std", "_mean": "mean"}


def space_names(dataset: str, variant: str) -> list[str]:
    cfg = config.load(dataset)
    names = [n + variant if (variant and n in descriptors.ATOMIC) else n for n in cfg["descriptors"]]
    return [n for n in names if n in descriptors.available(dataset)]


def knn_sets(D: np.ndarray, k: int) -> np.ndarray:
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1, kind="stable")[:, :k]


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan


def geometry(dataset: str, names: list[str], k: int = K_NEIGHBOURS) -> pd.DataFrame:
    X = {n: descriptors.compute(dataset, n) for n in names}
    common = np.all([np.isfinite(X[n]).all(axis=1) for n in names], axis=0)
    dvec = {n: pdist(descriptors.standardize(X[n][common])) for n in names}
    nn = {n: knn_sets(squareform(dvec[n]), k) for n in names}
    rows = []
    for a, b in itertools.product(names, names):
        rho = 1.0 if a == b else spearmanr(dvec[a], dvec[b])[0]
        knn = 1.0 if a == b else float(np.mean([jaccard(nn[a][i], nn[b][i]) for i in range(len(nn[a]))]))
        rows.append(dict(space_a=a, space_b=b, rank_agreement=rho, knn_overlap=knn,
                         n_frames=int(common.sum()), k=k))
    return pd.DataFrame(rows)


def cut_overlap(dataset: str, names: list[str], sampler: str) -> pd.DataFrame:
    sels = {n: selection.load(dataset, selection.method_id(sampler, n)) for n in names}
    rnd = selection.load(dataset, "random")

    def cuts(sel):
        out = {}
        for p, r, idx in selection.iter_cuts(sel):
            out[(p, r)] = idx
        return out

    C = {n: cuts(s) for n, s in sels.items()}
    R = cuts(rnd)
    n_rep = len(rnd["replicates"])
    rows = []
    for p in rnd["percents"]:
        # different spaces are compared at different seeds (r vs r+1), like the same-space and
        # random references: with the same seed, FPS starts from the same random frame in
        # every space, which would inflate the overlap of the smallest cuts
        pair_means = [np.mean([jaccard(C[a][(p, r)], C[b][(p, (r + 1) % n_rep)]) for r in range(n_rep)])
                      for a, b in itertools.combinations(names, 2)]
        same = np.mean([jaccard(C[n][(p, r)], C[n][(p, (r + 1) % n_rep)]) for n in names for r in range(n_rep)])
        rand = np.mean([jaccard(R[(p, r)], R[(p, (r + 1) % n_rep)]) for r in range(n_rep)])
        rows.append(dict(sampler=sampler, percent=p, between_mean=float(np.mean(pair_means)),
                         between_min=float(np.min(pair_means)), between_max=float(np.max(pair_means)),
                         same_space=float(same), random=float(rand), n_spaces=len(names)))
    return pd.DataFrame(rows)


def run(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    geo, ov = [], []
    for variant, agg in VARIANTS.items():
        names = space_names(dataset, variant)
        g = geometry(dataset, names)
        g.insert(0, "aggregation", agg)
        geo.append(g)
        for s in selection.SAMPLERS:
            try:
                o = cut_overlap(dataset, names, s)
            except FileNotFoundError:
                continue
            o.insert(0, "aggregation", agg)
            ov.append(o)
        off = g[g.space_a != g.space_b]
        print(f"[{dataset}] cross-space ({agg}): rank agreement {off.rank_agreement.mean():.2f} "
              f"({off.rank_agreement.min():.2f}-{off.rank_agreement.max():.2f}), "
              f"kNN overlap {off.knn_overlap.mean():.2f} ({off.knn_overlap.min():.2f}-{off.knn_overlap.max():.2f})")
    geo, ov = pd.concat(geo, ignore_index=True), pd.concat(ov, ignore_index=True)
    A = paths.results_dir(dataset, "analysis")
    geo.to_csv(A / "crossspace_geometry.csv", index=False)
    ov.to_csv(A / "cut_overlap.csv", index=False)
    return geo, ov
