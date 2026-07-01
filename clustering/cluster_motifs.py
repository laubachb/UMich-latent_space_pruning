"""
Unsupervised clustering of each descriptor space and its agreement with physics.

For every cached descriptor space we run KMeans and Ward agglomerative clustering
(k = number of metadata groups) on the standardized descriptors, then score how
well the discovered clusters recover the metadata `group` labels:

  * ARI / AMI                 — chance-corrected label agreement
  * homogeneity / completeness / v-measure
  * silhouette                — intrinsic cluster quality (no labels used)

A high ARI means the descriptor's clustering motifs coincide with the physical
origin of the frames (i.e. the latent space "discovers" the physics). We also
plot the composition of each discovered cluster to read off which motifs form.

Run from this directory (after compute_descriptor_matrices.py):
    python cluster_motifs.py

Outputs (figures/):
    clustering_metrics_comparison.png
    clustering_cluster_composition.png
Plus clustering/cache/cluster_metrics.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
    silhouette_score,
)

import common

SCORE_KEYS = ["ARI", "AMI", "homogeneity", "completeness", "v_measure", "silhouette"]


def score_clustering(X, pred, truth):
    hom, comp, vme = homogeneity_completeness_v_measure(truth, pred)
    sil = silhouette_score(X, pred) if len(set(pred)) > 1 else np.nan
    return {
        "ARI": adjusted_rand_score(truth, pred),
        "AMI": adjusted_mutual_info_score(truth, pred),
        "homogeneity": hom,
        "completeness": comp,
        "v_measure": vme,
        "silhouette": sil,
    }


def main():
    meta = pd.read_csv(common.CACHE_DIR / "metadata.csv")
    descriptors = common.available_descriptors()
    if not descriptors:
        raise SystemExit("No cached descriptors. Run compute_descriptor_matrices.py first.")

    truth = meta["group"].to_numpy()
    k = len(set(truth))
    print(f"Clustering into k={k} clusters (= number of metadata groups); "
          f"descriptors: {descriptors}")

    rows = []
    best = {"algo": None, "descriptor": None, "ARI": -np.inf, "pred": None}
    for name in descriptors:
        X = common.standardize(common.load_descriptor_matrix(name))
        algos = {
            "kmeans": KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X),
            "ward": AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X),
        }
        for algo, pred in algos.items():
            s = score_clustering(X, pred, truth)
            s.update({"descriptor": name, "algorithm": algo})
            rows.append(s)
            print(f"  {name:12s} {algo:7s} ARI={s['ARI']:.3f} AMI={s['AMI']:.3f} "
                  f"sil={s['silhouette']:.3f}")
            if s["ARI"] > best["ARI"]:
                best.update({"algo": algo, "descriptor": name, "ARI": s["ARI"], "pred": pred})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(common.CACHE_DIR / "cluster_metrics.csv", index=False)

    # ── Figure 1: metric comparison across descriptors (KMeans) ──────────────
    km = metrics[metrics.algorithm == "kmeans"].set_index("descriptor")
    fig, ax = plt.subplots(figsize=(1.6 * len(descriptors) + 4, 5))
    x = np.arange(len(descriptors))
    width = 0.8 / len(SCORE_KEYS)
    for i, key in enumerate(SCORE_KEYS):
        ax.bar(x + (i - len(SCORE_KEYS) / 2) * width, km.loc[descriptors, key].values,
               width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(descriptors, rotation=15, ha="right")
    ax.set_ylabel("score")
    ax.set_title(f"Cluster–metadata agreement (KMeans, k={k})",
                 fontsize=13, fontweight="bold")
    ax.axhline(0, c="k", lw=0.8)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    out1 = common.FIGURES_DIR / "clustering_metrics_comparison.png"
    fig.savefig(out1, dpi=250, bbox_inches="tight")
    print(f"Saved: {out1}")

    # ── Figure 2: composition of discovered clusters (best descriptor) ───────
    ct = pd.crosstab(best["pred"], truth)
    ct = ct.div(ct.sum(axis=1), axis=0)  # row-normalize -> composition
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(ct))
    for j, g in enumerate(ct.columns):
        ax.bar(ct.index.astype(str), ct[g].values, bottom=bottom,
               label=str(g), color=common.DESCRIPTOR_COLORS.get(g, None)
               or plt.cm.tab10(j % 10))
        bottom += ct[g].values
    ax.set_xlabel("discovered cluster")
    ax.set_ylabel("metadata-group composition")
    ax.set_title(f"Cluster composition — {best['descriptor']} ({best['algo']}, "
                 f"ARI={best['ARI']:.2f})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out2 = common.FIGURES_DIR / "clustering_cluster_composition.png"
    fig.savefig(out2, dpi=250, bbox_inches="tight")
    print(f"Saved: {out2}")

    print("\n" + "=" * 70)
    print("CLUSTER–METADATA AGREEMENT (sorted by ARI)")
    print("=" * 70)
    print(metrics.sort_values("ARI", ascending=False)
          [["descriptor", "algorithm"] + SCORE_KEYS].round(3).to_string(index=False))
    print(f"\nBest: {best['descriptor']} ({best['algo']}) ARI={best['ARI']:.3f}")


if __name__ == "__main__":
    main()
