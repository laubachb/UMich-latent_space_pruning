#!/usr/bin/env python3
"""Information content of the subsets relative to random selection, by latent space.

Breaks down the manuscript's information figure (fig_information) by latent space.
Rows: QUESTS diversity D(S)/D(full), entropy H(S)/H(full) and coverage, each as the
difference from random subsets of the same size. Columns: FPS, k-medoids, leverage.
For one element, latent space and retention, the difference is the median over the 25
replicates of the sampler minus the median over the 25 random replicates (the
definition that reproduces the published figure exactly). Coloured lines are the median
of that difference over the six elements for each latent space; the black line is the
median over all 36 element / latent-space combinations, i.e. the published curve.

    python plotting/fig_information_by_space.py   # -> figures/information/fig_information_by_space.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "information"
ELEMENTS = ["Ni", "Cu", "Li", "Mo", "Si", "Ge"]
SPACES = [("soap_mean", "SOAP", "#3A86FF"), ("behler_mean", "ACSF", "#06A77D"),
          ("bispectrum_mean", "Bispectrum", "#F77F00"), ("chimes", "ChIMES A-matrix", "#E63946"),
          ("chimes_fingerprint", "ChIMES fingerprint", "#8B1E3F"), ("euler", "Euler", "#7F7F00")]
SAMPLERS = [("fps", "FPS"), ("kmedoids", "k-medoids"), ("leverage", "Leverage")]
METRICS = [("D_rel", "Δ Diversity vs random"), ("H_rel", "Δ Entropy vs random"), ("coverage", "Δ Coverage vs random")]


def differences() -> pd.DataFrame:
    rows = []
    for el in ELEMENTS:
        q = pd.read_csv(ROOT / "results" / el / "analysis" / "quests.csv")
        med = q.groupby(["method", "percent"])[[m for m, _ in METRICS]].median()
        rnd = med.loc["random"]
        for s, _ in SAMPLERS:
            for sp, _, _ in SPACES:
                d = (med.loc[f"{s}-{sp}"] - rnd).reset_index()
                d["element"], d["sampler"], d["space"] = el, s, sp
                rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    D = differences()
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 11.0), sharex=True, sharey="row")
    letters = iter("abcdefghi")
    for r, (m, ylab) in enumerate(METRICS):
        for c, (s, sname) in enumerate(SAMPLERS):
            ax = axes[r, c]
            sub = D[D.sampler == s]
            for sp, label, col in SPACES:
                g = sub[sub.space == sp].groupby("percent")[m].median()
                ax.plot(g.index, g.values, color=col, lw=1.8, label=label)
            pooled = sub.groupby("percent")[m].median()
            ax.plot(pooled.index, pooled.values, color="black", lw=2.6, label="All spaces (median)")
            ax.axhline(0, color="black", ls=":", lw=1.0)
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 5, 10, 20, 50, 90]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "90"])
            ax.minorticks_off()
            ax.text(0.0, 1.03, f"({next(letters)})", transform=ax.transAxes, fontsize=14)
            if r == 0:
                ax.set_title(sname, fontsize=15, fontweight="bold")
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 2:
                ax.set_xlabel("% retained")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="black", ls=":", lw=1.0)); labels.append("Parity with random")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=12.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(OUT / f"fig_information_by_space.{ext}", **kw)
    print(f"wrote {OUT}/fig_information_by_space.pdf/.png")
    low = D[D.percent <= 10].groupby(["sampler", "space"])[[m for m, _ in METRICS]].median().round(3)
    print("median difference from random over 1-10 % retention and the six elements:")
    print(low.to_string())


if __name__ == "__main__":
    main()
