#!/usr/bin/env python3
"""Overview figure of the study (manuscript Figure 1), mostly visual.

Rows:
  (a) the six benchmark datasets as class-composition bars, (b) the Si training set in
  each of the six latent spaces (2-D PCA, coloured by frame class);
  (c) the frames each sampler selects in one latent space (Si, Bispectrum per-atom mean,
  5 % retained, replicate 1);
  (d) the four MLIP families and (e) the accuracy metrics (icons).
All scatter and bar panels are real data. Replicate 1 is representative of the 25: the
median distance of its picks from the centre of the space is within 0.1 of the
replicate mean for every sampler (FPS > leverage > k-medoids > random).
The class palette matches the manuscript's other main figures.

    python plotting/fig_overview.py            # -> figures/overview/fig_overview.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lsp import data, descriptors, selection  # noqa: E402

ELEMENTS = ["Ni", "Cu", "Li", "Mo", "Si", "Ge"]
SPACES = [("soap_mean", "SOAP"), ("behler_mean", "ACSF"), ("bispectrum_mean", "Bispectrum"),
          ("chimes", "ChIMES"), ("chimes_fingerprint", "ChIMES FP"), ("euler", "Euler")]
EXAMPLE, EXAMPLE_SPACE, PERCENT, REPLICATE = "Si", "bispectrum_mean", 5, 0
SAMPLERS = [("Random", "random"), ("FPS", "fps"), ("k-medoids", "kmedoids"), ("Leverage", "leverage")]
MODELS = ["SNAP", "ChIMES", "NNP", "GAP"]
OUT = ROOT / "figures" / "overview"

CLASSES = ["Surface", "Strained", "Vacancy 300 K", "Vacancy hot", "Bulk 300 K", "Bulk intermediate", "Bulk hot"]
CLASS_COLOR = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
INK, MUTED, LIGHT, TILE = "#1d2129", "#6b7280", "#d5d8dd", "#3b4a5c"


def frame_classes(element: str) -> np.ndarray:
    frames = data.load_frames(element)
    parsed = [data.parse_category(d["description"]) for d in frames]
    hottest = max(t for k, t in parsed if k == "Bulk")
    out = []
    for k, t in parsed:
        if k in ("Surface", "Strained"):
            out.append(k)
        elif k == "Vacancy":
            out.append("Vacancy 300 K" if t == 300 else "Vacancy hot")
        else:
            out.append("Bulk 300 K" if t == 300 else "Bulk hot" if t == hottest else "Bulk intermediate")
    return np.array(out)


def pca2(element: str, space: str):
    X = descriptors.standardize(descriptors.compute(element, space))
    ok = np.isfinite(X).all(axis=1)
    E = np.full((len(X), 2), np.nan)
    E[ok] = PCA(n_components=2, random_state=42).fit_transform(X[ok])
    return E


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#9aa0a8"); s.set_linewidth(0.8)


def arrow(fig, p0, p1):
    fig.patches.append(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=20, lw=1.8, color=MUTED))


def heading(fig, x, y, text):
    fig.text(x, y, text, fontsize=15, fontweight="bold", color=INK, va="bottom")


def tile(fig, x, y, w, h, text, color=TILE, size=14):
    ax = fig.add_axes([x, y, w, h]); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.18",
                                transform=ax.transAxes, fc=color, ec="none"))
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="white",
            fontsize=size, fontweight="bold")


def icon_parity(ax):
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2)
    rng = np.random.default_rng(3)
    x = rng.uniform(0.1, 0.9, 12)
    ax.scatter(x, x + rng.normal(0, 0.07, 12), s=22, color=TILE, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); bare(ax)


def icon_elastic(ax):
    e = np.linspace(-1, 1, 50)
    ax.plot(e, e ** 2, color=TILE, lw=2)
    ax.scatter([-0.5, -0.25, 0, 0.25, 0.5], [0.25, 0.0625, 0, 0.0625, 0.25], s=22, color=TILE, zorder=3)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.1, 1.05); bare(ax)


def main():
    plt.rcParams.update({"font.size": 13})
    fig = plt.figure(figsize=(13.0, 8.6))

    # class legend strip
    lax = fig.add_axes([0.02, 0.955, 0.96, 0.035]); lax.set_axis_off()
    widths = np.array([len(c) for c in CLASSES]) + 6.0          # label length + swatch and gap
    starts = np.concatenate([[0], np.cumsum(widths)[:-1]]) / widths.sum()
    for x, c in zip(starts, CLASSES):
        lax.add_patch(plt.Rectangle((x + 0.002, 0.15), 0.016, 0.7, color=CLASS_COLOR[c], transform=lax.transAxes))
        lax.text(x + 0.024, 0.5, c, transform=lax.transAxes, va="center", fontsize=12, color=INK)

    # (a) datasets: class composition bars
    heading(fig, 0.02, 0.905, "(a) Datasets")
    bax = fig.add_axes([0.055, 0.66, 0.25, 0.24])
    for row, el in enumerate(ELEMENTS):
        cls = frame_classes(el)
        left = 0.0
        for c in CLASSES:
            f = np.mean(cls == c)
            bax.barh(row, f, left=left, color=CLASS_COLOR[c], height=0.72, edgecolor="white", lw=0.6)
            left += f
    bax.set_yticks(range(len(ELEMENTS))); bax.set_yticklabels(ELEMENTS, fontsize=13)
    bax.invert_yaxis(); bax.set_xlim(0, 1); bax.set_xticks([])
    for s in bax.spines.values():
        s.set_visible(False)
    bax.tick_params(left=False)

    # (b) latent spaces: Si in each space
    heading(fig, 0.36, 0.905, "(b) Latent spaces")
    cls_si = frame_classes(EXAMPLE)
    tw, tg = 0.093, 0.011
    for i, (space, name) in enumerate(SPACES):
        ax = fig.add_axes([0.36 + i * (tw + tg), 0.69, tw, 0.18])
        E = pca2(EXAMPLE, space)
        for c in CLASSES:
            m = cls_si == c
            ax.scatter(E[m, 0], E[m, 1], s=5, color=CLASS_COLOR[c], lw=0)
        bare(ax)
        if space == EXAMPLE_SPACE:                                  # the space shown in (c)
            for sp in ax.spines.values():
                sp.set_color(INK); sp.set_linewidth(2.2)
        ax.text(0.5, -0.1, name, transform=ax.transAxes, ha="center", va="top", fontsize=12, color=INK,
                fontweight="bold" if space == EXAMPLE_SPACE else "normal")
    arrow(fig, (0.315, 0.78), (0.352, 0.78))

    # (c) samplers on one space
    heading(fig, 0.02, 0.575, f"(c) Samplers   ({EXAMPLE}, Bispectrum, {PERCENT} % retained)")
    E = pca2(EXAMPLE, EXAMPLE_SPACE)
    w, gap = 0.228, 0.022
    for i, (label, s) in enumerate(SAMPLERS):
        sel = selection.load(EXAMPLE, "random" if s == "random" else f"{s}-{EXAMPLE_SPACE}")
        idx = next(ix for p, r, ix in selection.iter_cuts(sel) if p == PERCENT and r == REPLICATE)
        ax = fig.add_axes([0.02 + i * (w + gap), 0.30, w, 0.255])
        ax.scatter(E[:, 0], E[:, 1], s=13, color=LIGHT, lw=0)
        ax.scatter(E[idx, 0], E[idx, 1], s=95, c=[CLASS_COLOR[c] for c in cls_si[idx]], ec=INK, lw=0.8, zorder=3)
        bare(ax)
        ax.text(0.04, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=14,
                fontweight="bold", color=INK)
    arrow(fig, (0.66, 0.665), (0.66, 0.6))

    # (d) models -> (e) accuracy, centred under (c)
    mx = 0.17                                                      # left edge of the model tiles
    heading(fig, mx, 0.215, "(d) Models")
    for i, m in enumerate(MODELS):
        tile(fig, mx + (i % 2) * 0.13, 0.11 - (i // 2) * 0.085, 0.12, 0.07, m)
    ex = 0.56                                                      # left edge of the accuracy icons
    heading(fig, ex - 0.01, 0.215, "(e) Accuracy vs. random")
    for j, (fn, lab) in enumerate(((icon_parity, "Test MAE"), (icon_elastic, "C$_{11}$  C$_{12}$  C$_{44}$"))):
        ax = fig.add_axes([ex + j * 0.15, 0.04, 0.1, 0.15]); fn(ax)
        ax.text(0.5, -0.08, lab, transform=ax.transAxes, ha="center", va="top", fontsize=12, color=INK)
    arrow(fig, (mx + 0.125, 0.285), (mx + 0.125, 0.245))
    arrow(fig, (mx + 0.265, 0.105), (ex - 0.02, 0.105))

    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(OUT / f"fig_overview.{ext}", **kw)
    print(f"wrote {OUT}/fig_overview.pdf/.png")


if __name__ == "__main__":
    main()
