"""
Per-state-point breakdown of the max-normalized interpoint-distance distribution,
one bulk figure per latent space.

For each cached descriptor space this makes a 3x3 grid of subplots, one per
state-point category. Every subplot draws the SAME bulk distance curve (the
max-normalized distribution of all pairwise distances in that space, as a light
gray reference outline) and fills in the slice contributed by that one category.

How the slice is defined (a true partition of the bulk):
  * standardize the raw (n_frames, n_features) matrix (per-feature z-score) and
    form the full pairwise euclidean distance matrix;
  * divide by that space's OWN maximum distance so the support is [0, 1], exactly
    as in normalized_distance_histograms.py;
  * treat every DIRECTED distance i->j as "owned" by the category of its source
    frame i. Each of the N*(N-1) directed distances has exactly one source, so
    summing the nine per-category histograms reproduces the bulk curve — the
    colored fills stack to the whole. The area of a category's fill equals that
    category's fraction of the frames.

Read it as: "when you stand on this state point and look out at the rest of the
dataset, where do those distances land?" A fill piled near 0 means the category
sits in the crowded center of the space; a fill pushed toward 1 means it is an
outlier that helps set the scale.

Run from the clustering/ directory (after compute_descriptor_matrices.py):
    python distance_by_category.py

Outputs (one per cached space):
    figures/clustering_distance_by_category_<descriptor>.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist

import common

N_BINS = 50

# 9-category, temperature-resolved state-point scheme, matching
# pruning/normalized_category_composition_by_temperature.py (colors included so
# a category is the same hue in every figure).
BULK_TEMPS = ["300", "843", "1518", "2530", "3374"]

CATEGORY_ORDER = [
    "Strained", "Surface",
    "Vacancy (300K)", "Vacancy (3374K)",
    "Bulk (300K)", "Bulk (843K)", "Bulk (1518K)", "Bulk (2530K)", "Bulk (3374K)",
]

CATEGORY_COLORS = {
    "Strained":        "#238B45",
    "Surface":         "#DD3497",
    "Vacancy (300K)":  "#9ECAE1",
    "Vacancy (3374K)": "#2171B5",
    "Bulk (300K)":     "#FEE0B6",
    "Bulk (843K)":     "#FDC171",
    "Bulk (1518K)":    "#FB9A3F",
    "Bulk (2530K)":    "#EC7014",
    "Bulk (3374K)":    "#C24E02",
}


def categorize(description: str) -> str:
    desc = description.lower()
    if "strain" in desc or "mode" in desc:
        return "Strained"
    if "surface" in desc:
        return "Surface"
    if "vacancy" in desc:
        return "Vacancy (3374K)" if "3374 K" in description else "Vacancy (300K)"
    for temp in BULK_TEMPS:
        if f"{temp} K" in description:
            return f"Bulk ({temp}K)"
    return "Bulk (300K)"  # 'Ground state crystal' -> T=0 endpoint of the ladder


def normalized_distance_matrix(matrix):
    """Standardized euclidean distances / own max -> off-diagonal support [0, 1]."""
    X = common.standardize(matrix)
    D = squareform(pdist(X, metric="euclidean"))
    dmax = D[~np.eye(len(D), dtype=bool)].max()
    if dmax > 0:
        D = D / dmax
    np.fill_diagonal(D, np.nan)  # drop self-distances from every histogram
    return D


def make_figure(name, categories):
    D = normalized_distance_matrix(common.load_descriptor_matrix(name))
    n = len(D)
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_w = bins[1] - bins[0]
    total_directed = n * (n - 1)  # all directed off-diagonal distances

    # Bulk density (all directed distances). Because the per-category source
    # partition is exact, sum(category densities) == this pointwise.
    bulk_counts, _ = np.histogram(D[~np.isnan(D)], bins=bins)
    bulk_density = bulk_counts / (total_directed * bin_w)

    # Per-category density on the SAME normalization (owned by source frame).
    cat_density = {}
    for cat in CATEGORY_ORDER:
        rows = np.where(categories == cat)[0]
        vals = D[rows, :]
        vals = vals[~np.isnan(vals)]
        counts, _ = np.histogram(vals, bins=bins)
        cat_density[cat] = counts / (total_directed * bin_w)

    y_max = bulk_density.max() * 1.08
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.5), sharex=True, sharey=True,
                             layout="constrained")

    for ax, cat in zip(axes.flat, CATEGORY_ORDER):
        count = int((categories == cat).sum())
        frac = count / n
        # Light gray reference outline of the full distribution.
        ax.stairs(bulk_density, bins, fill=True, color="0.90",
                  edgecolor="0.55", linewidth=1.0, zorder=1)
        # This category's slice, filled in its state-point color.
        edge = "#333333"
        ax.stairs(cat_density[cat], bins, fill=True, color=CATEGORY_COLORS[cat],
                  edgecolor=edge, linewidth=1.0, alpha=0.95, zorder=2)
        ax.set_title(f"{cat}   (n={count}, {frac*100:.0f}%)",
                     fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_max)
        ax.grid(False)

    fig.supxlabel("Interpoint Distance / Max Distance", fontsize=13, fontweight="bold")
    fig.supylabel("Probability Density", fontsize=13, fontweight="bold")
    fig.suptitle(
        f"{common.descriptor_label(name)} latent space — interpoint-distance "
        f"distribution by state point\n"
        f"(gray = all state points; color = this category's contribution, "
        f"fills stack to the bulk)",
        fontsize=13, fontweight="bold",
    )

    out = common.FIGURES_DIR / f"clustering_distance_by_category_{name}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    descriptors = common.available_descriptors()
    if not descriptors:
        raise SystemExit("No cached descriptors. Run compute_descriptor_matrices.py first.")
    print(f"Descriptor spaces: {descriptors}")

    data = common.load_frames()
    categories = np.array([categorize(d.get("description", "")) for d in data])

    # Report the state-point census once (shared across all spaces).
    print("\nState-point census (of %d frames):" % len(data))
    for cat in CATEGORY_ORDER:
        print(f"  {cat:16s} {int((categories == cat).sum()):4d}")
    missing = sorted(set(categories) - set(CATEGORY_ORDER))
    if missing:
        raise SystemExit(f"Uncategorized descriptions mapped to: {missing}")

    for name in descriptors:
        make_figure(name, categories)


if __name__ == "__main__":
    main()
