"""2D embeddings (UMAP and PCA) of every latent space, for the figures.

UMAP: n_neighbors 15, min_dist 0.9 (spreads the tight AIMD clusters so overlapping
categories stay visible), seed 42. PCA: first two components of the standardised
space; the explained variance is reported. Frames with an undefined descriptor
(NaN rows) get NaN coordinates.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .. import data, descriptors, paths

UMAP_PARAMS = dict(n_components=2, n_neighbors=15, min_dist=0.9, random_state=42)


def run(dataset: str, names: list[str] | None = None) -> pd.DataFrame:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import umap
    from sklearn.decomposition import PCA
    frames = data.load_frames(dataset)
    cats, cat = data.categories(frames)
    names = names or descriptors.available(dataset)
    rows = []
    for name in names:
        X = descriptors.compute(dataset, name)
        ok = np.isfinite(X).all(axis=1)
        Z = descriptors.standardize(X[ok])
        emb = np.full((len(X), 2), np.nan)
        emb[ok] = umap.UMAP(**UMAP_PARAMS).fit_transform(Z)
        pca = PCA(n_components=2, random_state=42).fit(Z)
        pc = np.full((len(X), 2), np.nan)
        pc[ok] = pca.transform(Z)
        ev = float(pca.explained_variance_ratio_.sum())
        for i in range(len(X)):
            rows.append(dict(descriptor=name, frame=i, category_id=int(cat[i]), category=cats[cat[i]],
                             umap1=emb[i, 0], umap2=emb[i, 1], pc1=pc[i, 0], pc2=pc[i, 1], pca_explained=ev))
        print(f"[{dataset}] embeddings '{name}' (PCA 2 components explain {ev:.2f})")
    df = pd.DataFrame(rows)
    df.to_csv(paths.results_dir(dataset, "analysis") / "embeddings.csv", index=False)
    return df
