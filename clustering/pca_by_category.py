"""
PCA of each descriptor latent space, colored by the nine state-point categories.

For each of the four pruning-study descriptor spaces (Behler, Bispectrum, SOAP,
ChIMES) this standardizes the raw (n_frames, n_features) matrix (per-feature
z-score, the package convention) and projects it onto its own first two
principal components. Points are colored by state point using the same
9-category, temperature-resolved scheme (and colors) as the composition and
distance-by-category figures.

Each space gets its own PCA, so PC1/PC2 and the explained-variance percentages
(shown in each subplot title) are not comparable across panels — only the
within-panel geometry (which state points separate, which overlap) is.

Run from the clustering/ directory (after compute_descriptor_matrices.py):
    python pca_by_category.py

Output: figures/clustering_pca_by_category.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import common
from distance_by_category import categorize, CATEGORY_ORDER, CATEGORY_COLORS

# Four descriptor spaces -> 2x2 grid (matches the pruning-study set; euler,
# which is clustering-only, is intentionally left out of the 2x2).
PANEL_DESCRIPTORS = ["behler", "bispectrum", "soap", "chimes"]


def main():
    available = set(common.available_descriptors())
    descriptors = [d for d in PANEL_DESCRIPTORS if d in available]
    missing = [d for d in PANEL_DESCRIPTORS if d not in available]
    if missing:
        print(f"WARNING: no cached matrix for {missing}; those panels will be blank.")

    data = common.load_frames()
    categories = np.array([categorize(d.get("description", "")) for d in data])

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")

    for ax, name in zip(axes.flat, PANEL_DESCRIPTORS):
        if name not in available:
            ax.axis("off")
            continue

        X = common.standardize(common.load_descriptor_matrix(name))
        pca = PCA(n_components=2)
        Y = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_ * 100

        # Draw larger categories first so small ones (e.g. Surface) stay visible.
        order = sorted(CATEGORY_ORDER,
                       key=lambda c: -(categories == c).sum())
        for cat in order:
            mask = categories == cat
            ax.scatter(Y[mask, 0], Y[mask, 1], s=42,
                       color=CATEGORY_COLORS[cat], label=cat,
                       edgecolor="#333333", linewidth=0.4, alpha=0.9, zorder=2)

        ax.set_title(f"{common.descriptor_label(name)}   "
                     f"(PC1 {ev[0]:.0f}%, PC2 {ev[1]:.0f}%)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("PC1", fontsize=11, fontweight="bold")
        ax.set_ylabel("PC2", fontsize=11, fontweight="bold")
        ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
        ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)

    # One shared legend below, in fixed category order (not the per-panel z-order).
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=CATEGORY_COLORS[c], markeredgecolor="#333333",
                          markeredgewidth=0.4, markersize=9, label=c)
               for c in CATEGORY_ORDER]
    fig.legend(handles, CATEGORY_ORDER, title="State point", loc="outside lower center",
               ncol=5, fontsize=9, title_fontsize=11)

    fig.suptitle("Latent-space PCA colored by state point (each space z-scored, "
                 "own PCA)", fontsize=13, fontweight="bold")

    out = common.FIGURES_DIR / "clustering_pca_by_category.png"
    fig.savefig(out, dpi=200)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
