#!/usr/bin/env python3
"""Summary numbers of every dataset with results, as one JSON block for the results page.

    python scripts/artifact_data.py > /path/to/data.json

Per dataset: element, n_frames, categories with full-dataset shares, the figure tags
available, and per selection method: surface share and strained share at <= 10 %
retention (pooled), Spearman rho(mean distance, enrichment), QUESTS D_rel at 5 %.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lsp import config, data, descriptors, paths, selection  # noqa: E402

FIGS = ["fig_dataset_composition", "fig_dataset_information", "fig_distance_analysis"]


def coarse_share(row, cats, prefix):
    return float(sum(row[c] for c in cats if c.startswith(prefix)))


def dataset_block(ds: str) -> dict | None:
    A = paths.results_dir(ds, "analysis")
    if not (A / "composition.csv").exists():
        return None
    cfg = config.load(ds)
    frames = data.load_frames(ds)
    cats, _ = data.categories(frames)
    comp = pd.read_csv(A / "composition.csv")
    dsel = pd.read_csv(A / "distance_selection.csv") if (A / "distance_selection.csv").exists() else None
    q = pd.read_csv(A / "quests.csv") if (A / "quests.csv").exists() else None
    full = comp[(comp.method == "random") & (comp.percent == 100)].iloc[0]
    methods = {}
    for m in selection.available(ds):
        low = comp[(comp.method == m) & (comp.percent <= 10)]
        if not len(low):
            continue
        share = low[cats].mean()
        s, name = selection.split_method(m)
        rho = None
        if name and dsel is not None:
            d = dsel[(dsel.descriptor == name) & (dsel.sampler == s)]
            rho = None if not len(d) or pd.isna(d.spearman_rho.iloc[0]) else round(float(d.spearman_rho.iloc[0]), 2)
        d5 = None
        if q is not None:
            g = q[(q.method == m) & (q.percent == 5)]
            d5 = round(float(g.D_rel.mean()), 3) if len(g) else None
        methods[m] = [round(coarse_share(share, cats, "Surface"), 1), round(coarse_share(share, cats, "Strained"), 1), rho, d5]
    tags = sorted(p.name for p in paths.FIGURES.glob(f"{ds}*")
                  if p.is_dir() and all((p / f"{f}.png").exists() for f in FIGS)
                  and (p.name == ds or p.name.startswith(ds + "_")))
    spaces = [[n, descriptors.label(n), descriptors.COLORS.get(descriptors.split_name(n)[0], "#333")]
              for n in cfg["descriptors"] if n in descriptors.available(ds)]
    n_rep = len(selection.load(ds, "random")["replicates"])
    return dict(element=cfg["element"], n_frames=len(frames), n_replicates=n_rep, categories=cats,
                full=dict(surface=round(coarse_share(full, cats, "Surface"), 1),
                          strained=round(coarse_share(full, cats, "Strained"), 1),
                          D5random=methods.get("random", [None] * 4)[3]),
                spaces=spaces, tags=tags, methods=methods)


def main():
    out = {}
    for ds in config.available_datasets():
        if ds == "si_legacy":
            continue
        b = dataset_block(ds)
        if b:
            out[ds] = b
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
