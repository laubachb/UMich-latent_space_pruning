"""
Max-normalized interpoint-distance distributions for each descriptor latent space.

For every cached descriptor space:
  1. standardize the raw (n_frames, n_features) matrix (per-feature z-score, the
     same convention the rest of the clustering package uses);
  2. compute all pairwise euclidean distances between frames;
  3. divide by that space's OWN maximum distance so every distribution lives on a
     common [0, 1] support (nearest pair -> 0, farthest pair -> 1);
  4. overlay the histograms on a single axis.

Why max-normalization (vs. the mean-normalization in distance_analysis.py):
dividing by the max fixes the support to [0, 1] for all spaces, so the SHAPE of
the distribution — how frames are spread between the closest and farthest pair —
is directly comparable across descriptors even though the raw geometry differs.
A distribution piled up near 0 with a long thin tail means "a few outliers set
the scale, everything else is bunched"; a broad/centered distribution means the
frames fill the space more uniformly.

Run from this directory (after compute_descriptor_matrices.py):
    python normalized_distance_histograms.py

Outputs:
    figures/clustering_normalized_distance_histograms.png
    clustering/cache/normalized_distance_stats.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

import common

N_BINS = 60
STANDARDIZE = True  # match the rest of the package (raw matrices cached, z-scored here)


def max_normalized_distances(matrix):
    """Standardized euclidean pdist divided by its own maximum -> support [0, 1]."""
    X = common.standardize(matrix) if STANDARDIZE else np.asarray(matrix, float)
    d = pdist(X, metric="euclidean")
    dmax = d.max()
    return d / dmax if dmax > 0 else d


def distribution_stats(dnorm, bins):
    """Summary shape descriptors of a max-normalized distance distribution."""
    hist, _ = np.histogram(dnorm, bins=bins, range=(0.0, 1.0), density=True)
    p = hist / hist.sum()
    nz = p > 0
    # Shannon entropy of the (max-normalized) distance histogram, in bits, and
    # normalized to [0, 1] by log2(n_bins) so different bin counts are comparable.
    shannon_bits = float(-(p[nz] * np.log2(p[nz])).sum())
    return {
        "mean": float(dnorm.mean()),
        "median": float(np.median(dnorm)),
        "std": float(dnorm.std()),
        "skew_p10_p90": float(np.percentile(dnorm, 90) - np.percentile(dnorm, 10)),
        "distance_hist_entropy_bits": shannon_bits,
        "distance_hist_entropy_norm": shannon_bits / np.log2(len(hist)),
    }


def main():
    descriptors = common.available_descriptors()
    if not descriptors:
        raise SystemExit("No cached descriptors. Run compute_descriptor_matrices.py first.")
    print(f"Descriptor spaces: {descriptors}")

    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    dnorms, rows = {}, []
    for name in descriptors:
        dnorms[name] = max_normalized_distances(common.load_descriptor_matrix(name))
        stats = distribution_stats(dnorms[name], bins)
        stats["descriptor"] = name
        rows.append(stats)
        print(f"  {name:12s} mean={stats['mean']:.3f} median={stats['median']:.3f} "
              f"hist-entropy={stats['distance_hist_entropy_norm']:.3f}")

    # ── Figure: overlaid max-normalized distance histograms ──────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in descriptors:
        color = common.DESCRIPTOR_COLORS.get(name, None)
        ax.hist(dnorms[name], bins=bins, density=True, histtype="stepfilled",
                alpha=0.28, color=color)
        ax.hist(dnorms[name], bins=bins, density=True, histtype="step",
                linewidth=2.0, color=color, label=common.descriptor_label(name))
    ax.set_xlabel("Interpoint Distance / Max Distance  (Per Latent Space)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(title="Latent Space", fontsize=10)
    ax.grid(False)
    fig.tight_layout()
    out = common.FIGURES_DIR / "clustering_normalized_distance_histograms.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    print(f"Saved: {out}")

    stats_df = pd.DataFrame(rows).set_index("descriptor")
    stats_df.to_csv(common.CACHE_DIR / "normalized_distance_stats.csv")
    print("\n" + "=" * 70)
    print("MAX-NORMALIZED DISTANCE DISTRIBUTION SHAPE")
    print("=" * 70)
    print(stats_df.round(4).to_string())


if __name__ == "__main__":
    main()
