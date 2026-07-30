"""
Effective dimensionality (participation ratio) of each descriptor latent space.

The underlying frames are identical across descriptor spaces, so any difference in
effective dimensionality is a property of the REPRESENTATION, not the data. The
participation ratio of the PCA eigenvalue spectrum,

    PR = (sum_i lambda_i)^2 / sum_i lambda_i^2,

is the *effective number of independent directions* a representation uses to
encode the frames. PR = 1 means all variance sits on a single axis; PR = d means
variance is spread evenly across d axes. Higher PR => the representation spreads
the same data across more distinguishable directions (more informative).

Run from this directory (after compute_descriptor_matrices.py):
    python participation_ratio.py

Outputs:
    figures/clustering_participation_ratio.png
    clustering/cache/participation_ratio.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import common


def participation_ratio(matrix):
    """Effective number of independent PCA directions used by a latent space."""
    X = common.standardize(matrix)
    lam = np.clip(PCA(n_components=min(X.shape)).fit(X).explained_variance_, 0, None)
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def main():
    descriptors = common.available_descriptors()
    if not descriptors:
        raise SystemExit("No cached descriptors. Run compute_descriptor_matrices.py first.")

    rows = [{"descriptor": n, "participation_ratio": participation_ratio(
                common.load_descriptor_matrix(n))} for n in descriptors]
    df = pd.DataFrame(rows).sort_values("participation_ratio", ascending=False)
    df.to_csv(common.CACHE_DIR / "participation_ratio.csv", index=False)
    print(df.round(3).to_string(index=False))

    labels = [common.descriptor_label(d) for d in df["descriptor"]]
    colors = [common.DESCRIPTOR_COLORS.get(d, "#555") for d in df["descriptor"]]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(labels, df["participation_ratio"].values, color=colors,
                  edgecolor="k", linewidth=0.7, width=0.7)
    for bar, val in zip(bars, df["participation_ratio"].values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Participation Ratio (Effective Dimensionality)",
                  fontsize=12, fontweight="bold")
    ax.set_xlabel("Latent Space", fontsize=12, fontweight="bold")
    ax.set_ylim(0, df["participation_ratio"].max() * 1.15)
    ax.grid(False)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = common.FIGURES_DIR / "clustering_participation_ratio.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
