#!/usr/bin/env python3
"""Cross-dataset summary tables of the study.

    python scripts/summarize.py                 # all datasets with results
    python scripts/summarize.py Si Cu --sampler fps

Writes results/summary/<sampler>_summary.csv and results/summary/<sampler>_summary.md with,
per (dataset, latent space):
    n_features, participation_ratio          effective dimensionality of the space
    rho                                      Spearman rho(mean distance, enrichment <= 10 %)
    enrich_surface, enrich_strained, ...     enrichment of each coarse category at <= 10 % retention
    D_rel_5, D_rel_10, cov_5, cov_10         QUESTS diversity / coverage at 5 % and 10 % retention
                                             (mean over replicates; random baseline listed too)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lsp import config, descriptors, paths, selection  # noqa: E402

COARSE = {"Surface": "surface", "Strained": "strained", "Vacancy": "vacancy", "Bulk": "bulk"}


def coarse(cat: str) -> str:
    for k, v in COARSE.items():
        if cat.startswith(k):
            return v
    return "other"


def summarize(dataset: str, sampler: str) -> list[dict]:
    A = paths.results_dir(dataset, "analysis")
    if not (A / "composition.csv").exists():
        return []
    comp = pd.read_csv(A / "composition.csv")
    dim = pd.read_csv(A / "dimensionality.csv").set_index("descriptor") if (A / "dimensionality.csv").exists() else None
    dsel = pd.read_csv(A / "distance_selection.csv") if (A / "distance_selection.csv").exists() else None
    q = pd.read_csv(A / "quests.csv") if (A / "quests.csv").exists() else None
    cats = [c for c in comp.columns if c not in ("method", "percent", "n_pooled")]
    full = comp[(comp.method == "random") & (comp.percent == 100)].iloc[0]
    rows = []
    for m in ["random"] + [x for x in selection.available(dataset, sampler) if x != "random"]:
        s, name = selection.split_method(m)
        row = dict(dataset=dataset, sampler=s, space=name or "random", label=descriptors.label(name or "random"))
        if name and dim is not None and name in dim.index:
            row.update(n_features=int(dim.loc[name, "n_features"]),
                       participation_ratio=round(float(dim.loc[name, "participation_ratio"]), 2))
        if name and dsel is not None:
            d = dsel[(dsel.descriptor == name) & (dsel.sampler == s)]
            if len(d):
                row["rho"] = round(float(d.spearman_rho.iloc[0]), 2)
        low = comp[(comp.method == m) & (comp.percent <= 10)]
        if len(low):
            share = low[cats].mean()
            for cz in ("surface", "strained", "vacancy", "bulk"):
                cs = [c for c in cats if coarse(c) == cz]
                if cs and full[cs].sum() > 0:
                    row[f"enrich_{cz}"] = round(float(share[cs].sum() / full[cs].sum()), 2)
        if q is not None:
            for p in (5, 10):
                g = q[(q.method == m) & (q.percent == p)]
                if len(g):
                    row[f"D_rel_{p}"] = round(float(g.D_rel.mean()), 3)
                    row[f"cov_{p}"] = round(float(g.coverage.mean()), 3)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", default=[d for d in config.available_datasets() if d != "si_legacy"])
    ap.add_argument("--sampler", default="fps", choices=selection.SAMPLERS)
    a = ap.parse_args()
    rows = [r for ds in a.datasets for r in summarize(ds, a.sampler)]
    if not rows:
        sys.exit("no results found")
    df = pd.DataFrame(rows)
    out = paths.RESULTS / "summary"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"{a.sampler}_summary.csv", index=False)
    with open(out / f"{a.sampler}_summary.md", "w") as fh:
        fh.write(f"# {a.sampler}: per-dataset summary (enrichment and QUESTS metrics at low retention)\n\n")
        fh.write(df.drop(columns=["sampler"]).to_markdown(index=False))
        fh.write("\n")
    print(df.drop(columns=["sampler"]).to_string(index=False))
    print(f"\n-> {out / f'{a.sampler}_summary.csv'}")


if __name__ == "__main__":
    main()
