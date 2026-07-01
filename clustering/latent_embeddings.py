"""
2-D embeddings of each descriptor latent space, colored by metadata.

For every cached descriptor space we compute a PCA projection and a t-SNE
projection of the standardized descriptors, then color the points by:
  * group            (categorical physical origin: AIMD-NVT / Vacancy / ... )
  * temperature_K    (continuous)
  * energy_per_atom  (continuous)

This lets us eyeball whether physically-meaningful structure (thermal ladders,
strain families, vacancy vs bulk) organizes the latent space.

Run from this directory (after compute_descriptor_matrices.py):
    python latent_embeddings.py

Outputs (figures/):
    clustering_embedding_<descriptor>.png   one PCA+tSNE panel per descriptor
    clustering_pca_by_group.png             PCA-by-group across all descriptors
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import common

CONTINUOUS_COLORINGS = [("temperature_K", "T (K)", "plasma"),
                        ("energy_per_atom", "E/atom (eV)", "viridis")]
GROUP_PALETTE = ["#E63946", "#F77F00", "#06A77D", "#457B9D", "#A23B72",
                 "#8338EC", "#00B4D8", "#999999"]


def embed(matrix, seed=42):
    """Return (pca_2d, tsne_2d, explained_variance_ratio[:2])."""
    X = common.standardize(matrix)
    pca = PCA(n_components=min(2, X.shape[1]))
    pca_xy = pca.fit_transform(X)
    perplexity = min(30, max(5, (X.shape[0] - 1) // 3))
    tsne_xy = TSNE(n_components=2, init="pca", perplexity=perplexity,
                   random_state=seed).fit_transform(X)
    return pca_xy, tsne_xy, pca.explained_variance_ratio_[:2]


def scatter_categorical(ax, xy, labels, title):
    labels = np.asarray(labels)
    for i, g in enumerate(sorted(set(labels))):
        m = labels == g
        ax.scatter(xy[m, 0], xy[m, 1], s=18, alpha=0.8,
                   color=GROUP_PALETTE[i % len(GROUP_PALETTE)], label=str(g),
                   edgecolors="none")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.9)


def scatter_continuous(ax, xy, values, title, cmap):
    values = np.asarray(values, float)
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=18, alpha=0.85, c=values, cmap=cmap)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def main():
    meta = pd.read_csv(common.CACHE_DIR / "metadata.csv")
    descriptors = common.available_descriptors()
    if not descriptors:
        raise SystemExit("No cached descriptors. Run compute_descriptor_matrices.py first.")
    print(f"Embedding descriptor spaces: {descriptors}")

    embeddings = {}
    for name in descriptors:
        print(f"  embedding {name} ...")
        embeddings[name] = embed(common.load_descriptor_matrix(name))

    # ── Per-descriptor panel: PCA + tSNE x (group, T, E) ─────────────────────
    for name in descriptors:
        pca_xy, tsne_xy, evr = embeddings[name]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for row, (xy, tag) in enumerate([(pca_xy, "PCA"), (tsne_xy, "t-SNE")]):
            scatter_categorical(axes[row][0], xy, meta["group"].to_numpy(),
                                 f"{tag} — group")
            for col, (key, clabel, cmap) in enumerate(CONTINUOUS_COLORINGS, start=1):
                scatter_continuous(axes[row][col], xy, meta[key].to_numpy(),
                                   f"{tag} — {clabel}", cmap)
        var = f"PC1 {evr[0]*100:.0f}% · PC2 {evr[1]*100:.0f}%" if len(evr) > 1 else ""
        fig.suptitle(f"{name} latent space   [{var}]", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = common.FIGURES_DIR / f"clustering_embedding_{name}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig)

    # ── Cross-descriptor comparison: PCA-by-group side by side ───────────────
    ncols = len(descriptors)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.8), squeeze=False)
    for ax, name in zip(axes[0], descriptors):
        pca_xy, _, evr = embeddings[name]
        scatter_categorical(ax, pca_xy, meta["group"].to_numpy(), name)
        if len(evr) > 1:
            ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}%)")
            ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}%)")
    fig.suptitle("PCA of each descriptor space, colored by metadata group",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = common.FIGURES_DIR / "clustering_pca_by_group.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
