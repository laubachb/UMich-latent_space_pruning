#!/usr/bin/env python3
"""
Helpers for unsupervised baseline UMAPs: descriptor loading, UMAP, plotting.

Descriptors are NOT computed here. They are the structure-level latent spaces of the
manuscript, built and cached by the `lsp` package (results/<ds>/descriptors/<space>.npy;
computed on first use from configs/<ds>.json): the per-atom mean of SOAP,
Behler-Parrinello, bispectrum and MTP, the ChIMES A-matrix energy row / N, the
full-frame ChIMES fingerprint (alpha = 1) and the Euler characteristic curve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
UNSUP_DIR = SCRIPT_DIR.parent
UMAP_DIR = UNSUP_DIR.parent
ROOT_DIR = UMAP_DIR.parents[1]

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "analysis"))
from categorization import (  # noqa: E402
    CATEGORIES,
    PHASE_CATEGORIES,
)
from lsp import config as lsp_config  # noqa: E402
from lsp import data as lsp_data  # noqa: E402
from lsp import descriptors as lsp_desc  # noqa: E402

DATASET = "Si"
UMAP_COMMON = dict(n_components=2, min_dist=0.9, random_state=42)

CATEGORY_COLORS = {
    "Strained": "#E63946",
    "High Temp (3374K)": "#F77F00",
    "Normal (300K)": "#06A77D",
    "Vacancy (300K)": "#457B9D",
    "Surface": "#A23B72",
    "Other": "#999999",
}

PHASE_CATEGORY_COLORS = {
    "Vacancy-300K": "#457B9D",
    "Vacancy-3374K": "#F77F00",
    "Surface": "#A23B72",
    "Strained Diamond": "#C9A227",
    "Unstrained Diamond": "#06A77D",
    "Liquid": "#E63946",
}


def latent_spaces(dataset: str = DATASET) -> list[str]:
    """Latent-space ids in figure order, with the manuscript's per-atom-mean aggregation."""
    cfg = lsp_config.load(dataset)
    return [n + "_mean" if n in lsp_desc.ATOMIC else n for n in cfg["descriptors"]]


def load_frames(dataset: str = DATASET) -> list:
    """Frames of the dataset (mlearn training.json; see scripts/fetch_mlearn_data.py)."""
    return lsp_data.load_frames(dataset)


def load_descriptors(dataset: str = DATASET, force_recompute: bool = False) -> dict[str, np.ndarray]:
    """{panel title: raw (n_frames, n_features) matrix} for every latent space, from the lsp cache."""
    out = {}
    for name in latent_spaces(dataset):
        X = lsp_desc.compute(dataset, name, force=force_recompute)
        out[lsp_desc.label(lsp_desc.split_name(name)[0])] = X
        print(f"  {name}: {X.shape}")
    return out


def compute_umap_embedding(
    descriptors: np.ndarray,
    init: str = "spectral",
    min_dist: float | None = None,
    metric: str | None = None,
    n_neighbors: int | None = None,
    n_components: int | None = None,
    y: np.ndarray | None = None,
) -> np.ndarray:
    ok = np.isfinite(descriptors).all(axis=1)       # frames without a descriptor get NaN coordinates
    scaled = lsp_desc.standardize(descriptors[ok])
    params = {**UMAP_COMMON, "init": init}
    if min_dist is not None:
        params["min_dist"] = min_dist
    if metric is not None:
        params["metric"] = metric
    if n_neighbors is not None:
        params["n_neighbors"] = n_neighbors
    if n_components is not None:
        params["n_components"] = n_components
    reducer = umap.UMAP(**params)
    emb = np.full((len(descriptors), params["n_components"]), np.nan)
    emb[ok] = reducer.fit_transform(scaled) if y is None else reducer.fit_transform(scaled, y=np.asarray(y)[ok])
    return emb


def _panel_order(descriptor_dict: dict) -> list[str]:
    return list(descriptor_dict)                     # insertion order = latent_spaces()


def plot_descriptor_row(
    axes,
    embeddings: dict[str, np.ndarray],
    colors: np.ndarray,
    colorbar_label: str,
    titles: dict[str, str] | None = None,
    shared_colorbar: bool = True,
    cax=None,
):
    names = _panel_order(embeddings)
    sc = None
    for ax, name in zip(axes, names):
        sc = ax.scatter(
            embeddings[name][:, 0],
            embeddings[name][:, 1],
            c=colors,
            s=12,
            alpha=0.75,
            cmap="viridis",
        )
        ax.set_title(titles[name] if titles else name, fontsize=13)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    if sc is None:
        return
    if cax is not None:
        axes[0].figure.colorbar(sc, cax=cax, label=colorbar_label)
    elif shared_colorbar:
        fig = axes[0].figure
        fig.colorbar(sc, ax=list(axes), label=colorbar_label, fraction=0.02, pad=0.02)


def plot_category_row(
    axes,
    embeddings: dict[str, np.ndarray],
    categories: list[str],
    titles: dict[str, str] | None = None,
    legend_ax=None,
):
    names = _panel_order(embeddings)
    cats = np.asarray(categories)
    for ax, name in zip(axes, names):
        for cat in CATEGORIES:
            mask = cats == cat
            if not np.any(mask):
                continue
            ax.scatter(
                embeddings[name][mask, 0],
                embeddings[name][mask, 1],
                c=CATEGORY_COLORS[cat],
                s=18,
                alpha=0.8,
                label=cat,
                edgecolors="none",
            )
        ax.set_title(titles[name] if titles else name, fontsize=13)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CATEGORY_COLORS[cat],
            markersize=8,
            label=cat,
        )
        for cat in CATEGORIES
        if cat in cats
    ]
    lax = legend_ax if legend_ax is not None else axes[-1]
    lax.legend(handles=handles, loc="center" if legend_ax is not None else "best", fontsize=9,
               frameon=legend_ax is None, title="Category")


def plot_phase_category_row(
    axes,
    embeddings: dict[str, np.ndarray],
    phase_categories: list[str],
    titles: dict[str, str] | None = None,
    legend_ax=None,
):
    """Color by full phase categories."""
    names = _panel_order(embeddings)
    cats = np.asarray(phase_categories)
    for ax, name in zip(axes, names):
        for cat in PHASE_CATEGORIES:
            mask = cats == cat
            if not np.any(mask):
                continue
            ax.scatter(
                embeddings[name][mask, 0],
                embeddings[name][mask, 1],
                c=PHASE_CATEGORY_COLORS[cat],
                s=18,
                alpha=0.8,
                label=cat,
                edgecolors="none",
            )
        ax.set_title(titles[name] if titles else name, fontsize=13)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PHASE_CATEGORY_COLORS[cat],
            markersize=8,
            label=cat,
        )
        for cat in PHASE_CATEGORIES
        if cat in cats
    ]
    lax = legend_ax if legend_ax is not None else axes[-1]
    lax.legend(handles=handles, loc="center" if legend_ax is not None else "best", fontsize=9,
               frameon=legend_ax is None, title="Phase category")
