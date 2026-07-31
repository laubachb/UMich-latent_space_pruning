"""
Nominal-distance analysis of the descriptor latent spaces.

For every cached descriptor space this script asks two questions:

  1. Are metadata groups SEPARATED in the latent space?
     -> mean within-group vs between-group pairwise distance, and a
        group x group mean-distance heatmap.

  2. Does latent-space distance encode PHYSICS?
     -> Spearman correlation between pairwise descriptor distance and the
        pairwise difference in continuous physical properties
        (|dE/atom|, |dT|, |d(mean normal stress)|).

Distances are computed on the standardized descriptor matrix (euclidean) and
then divided by their global mean so different descriptor spaces are compared on
a common "nominal distance = 1" scale.

Run from this directory (after compute_descriptor_matrices.py):
    python distance_analysis.py

Outputs (figures/):
    clustering_group_distance_heatmaps.png
    clustering_within_between_separation.png
    clustering_distance_vs_physics.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import common

CATEGORICAL = ["group", "category"]
CONTINUOUS = [
    ("energy_per_atom", "|dE/atom| (eV)"),
    ("temperature_K", "|dT| (K)"),
    ("mean_normal_stress", "|d stress|"),
]


def normalized_distances(matrix):
    """Standardize -> euclidean pdist -> divide by global mean distance."""
    X = common.standardize(matrix)
    d = pdist(X, metric="euclidean")
    return d / d.mean(), squareform(d / d.mean())


def within_between(dmat, labels):
    """Mean within-group and between-group distance for a categorical label."""
    labels = np.asarray(labels)
    iu = np.triu_indices_from(dmat, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    within = dmat[iu][same]
    between = dmat[iu][~same]
    return (
        float(within.mean()) if within.size else np.nan,
        float(between.mean()) if between.size else np.nan,
    )


def group_distance_matrix(dmat, labels):
    """Block-mean distance between every pair of label values."""
    labels = np.asarray(labels)
    uniq = sorted(set(labels))
    M = np.full((len(uniq), len(uniq)), np.nan)
    for a, ga in enumerate(uniq):
        ia = np.where(labels == ga)[0]
        for b, gb in enumerate(uniq):
            ib = np.where(labels == gb)[0]
            block = dmat[np.ix_(ia, ib)]
            if ga == gb:
                mask = ~np.eye(len(ia), dtype=bool)
                vals = block[mask]
            else:
                vals = block.ravel()
            M[a, b] = vals.mean() if vals.size else np.nan
    return uniq, M


def physics_correlations(dvec, meta, iu):
    """Spearman r between descriptor pair-distance and |physical property diff|."""
    out = {}
    for key, label in CONTINUOUS:
        vals = meta[key].to_numpy(dtype=float)
        diff = np.abs(vals[iu[0]] - vals[iu[1]])
        ok = np.isfinite(diff) & np.isfinite(dvec)
        if ok.sum() > 10 and np.nanstd(diff[ok]) > 0:
            r, p = spearmanr(dvec[ok], diff[ok])
        else:
            r, p = np.nan, np.nan
        out[key] = (label, r, p, diff)
    return out


def main():
    meta = pd.read_csv(common.CACHE_DIR / "metadata.csv")
    descriptors = common.available_descriptors()
    if not descriptors:
        raise SystemExit("No cached descriptors. Run compute_descriptor_matrices.py first.")
    print(f"Analyzing descriptor spaces: {descriptors}")

    n = len(meta)
    iu = np.triu_indices(n, k=1)

    dvecs, dmats = {}, {}
    for name in descriptors:
        dvecs[name], dmats[name] = normalized_distances(common.load_descriptor_matrix(name))

    # ── Figure 1: group x group distance heatmaps ────────────────────────────
    ncols = len(descriptors)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.6), squeeze=False)
    for ax, name in zip(axes[0], descriptors):
        uniq, M = group_distance_matrix(dmats[name], meta["group"].to_numpy())
        im = ax.imshow(M, cmap="viridis")
        ax.set_xticks(range(len(uniq)))
        ax.set_yticks(range(len(uniq)))
        ax.set_xticklabels(uniq, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(uniq, fontsize=8)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        for i in range(len(uniq)):
            for j in range(len(uniq)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="w" if M[i, j] < np.nanmean(M) else "k", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Mean normalized latent-space distance between metadata groups",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out1 = common.FIGURES_DIR / "clustering_group_distance_heatmaps.png"
    fig.savefig(out1, dpi=250, bbox_inches="tight")
    print(f"Saved: {out1}")

    # ── Figure 2: within vs between separation ───────────────────────────────
    fig, axes = plt.subplots(1, len(CATEGORICAL), figsize=(7 * len(CATEGORICAL), 5),
                             squeeze=False)
    sep_table = {}
    for ax, label in zip(axes[0], CATEGORICAL):
        w = [within_between(dmats[nm], meta[label].to_numpy())[0] for nm in descriptors]
        b = [within_between(dmats[nm], meta[label].to_numpy())[1] for nm in descriptors]
        x = np.arange(len(descriptors))
        ax.bar(x - 0.2, w, 0.4, label="within-group", color="#457B9D")
        ax.bar(x + 0.2, b, 0.4, label="between-group", color="#E63946")
        ax.set_xticks(x)
        ax.set_xticklabels(descriptors, rotation=20, ha="right")
        ax.set_ylabel("mean normalized distance")
        ax.set_title(f"Separation by '{label}'", fontsize=12, fontweight="bold")
        ax.axhline(1.0, ls="--", c="gray", lw=1, alpha=0.7)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3, ls="--")
        sep_table[label] = {nm: (b[i] / w[i] if w[i] else np.nan)
                            for i, nm in enumerate(descriptors)}
    fig.suptitle("Within- vs between-group nominal distance "
                 "(ratio > 1 => groups are separated)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out2 = common.FIGURES_DIR / "clustering_within_between_separation.png"
    fig.savefig(out2, dpi=250, bbox_inches="tight")
    print(f"Saved: {out2}")

    # ── Figure 3: distance vs physics (Spearman) ─────────────────────────────
    fig, axes = plt.subplots(len(descriptors), len(CONTINUOUS),
                             figsize=(4.6 * len(CONTINUOUS), 3.8 * len(descriptors)),
                             squeeze=False)
    corr_rows = []
    rng = np.random.default_rng(0)
    for ri, name in enumerate(descriptors):
        corrs = physics_correlations(dvecs[name], meta, iu)
        for ci, (key, _lab) in enumerate(CONTINUOUS):
            ax = axes[ri][ci]
            label, r, p, diff = corrs[key]
            ok = np.isfinite(diff) & np.isfinite(dvecs[name])
            idx = np.where(ok)[0]
            if idx.size > 4000:  # subsample for a legible scatter
                idx = rng.choice(idx, 4000, replace=False)
            ax.scatter(diff[idx], dvecs[name][idx], s=4, alpha=0.15,
                       color=common.DESCRIPTOR_COLORS.get(name, "#333"))
            ax.set_xlabel(label, fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{name}\nnorm. distance", fontsize=10, fontweight="bold")
            title = f"Spearman r = {r:.2f}" if np.isfinite(r) else "n/a"
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.25, ls="--")
            corr_rows.append({"descriptor": name, "property": key, "spearman_r": r, "p": p})
    fig.suptitle("Does latent-space distance encode physics?\n"
                 "descriptor pair-distance vs |physical property difference|",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out3 = common.FIGURES_DIR / "clustering_distance_vs_physics.png"
    fig.savefig(out3, dpi=200, bbox_inches="tight")
    print(f"Saved: {out3}")

    # ── Text summary ─────────────────────────────────────────────────────────
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(common.CACHE_DIR / "distance_physics_correlations.csv", index=False)

    print("\n" + "=" * 70)
    print("BETWEEN/WITHIN SEPARATION RATIO (higher => metadata is well separated)")
    print("=" * 70)
    print(pd.DataFrame(sep_table).round(3).to_string())

    print("\n" + "=" * 70)
    print("SPEARMAN r: latent distance vs |physical difference|")
    print("=" * 70)
    print(corr_df.pivot(index="descriptor", columns="property",
                        values="spearman_r").round(3).to_string())


if __name__ == "__main__":
    main()
