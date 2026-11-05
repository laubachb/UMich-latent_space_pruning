"""Export the analysis tables as plain-text data for the gnuplot figures.

Writes plotting/data/<dataset>/:
    meta.gp                      gnuplot arrays: categories (names, colours, legend
                                 order) and methods (ids, names, colours, dash/point types)
    umap_<name>.dat              umap1 umap2 category_id           (one row per frame)
    toc_points_<name>.dat        pc1 pc2 category_id
    toc_selected_<name>.dat      pc1 pc2 of the FPS picks at 5 % (replicate 1)
    composition_<method>.dat     retention %, cumulative composition (%) per category
    information_<method>.dat     retention %, mean/sd of D_rel, coverage, H_rel
    distance_hist_<name>.dat     bin left edge, cumulative per-category density
    distance_selection_<name>.dat  category_id, mean distance, within distance, enrichment
    distance_rho.gp              array RHO[] of Spearman rho per latent space

One export per (aggregation variant, sampler): plotting/data/<dataset><agg><samp>/
with agg = "" (mean+std) or "_mean" (per-atom mean of the atom-centred spaces) and
samp = "" (FPS) or "_<sampler>" (kmedoids, leverage). The figures of that directory
compare random against "<sampler>-<latent space>" for every configured latent space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, data, descriptors, paths, selection

# category colours: Surface (magenta), vacancies (blues, cold -> hot), bulk (orange
# ramp, hot -> cold), strained (green) — the Si-study palette, extended as needed.
SURFACE_COL = "#E054A6"
STRAIN_COL = "#45A05A"
VAC_COLS = ["#ABD1E3", "#3E86C2", "#1F4E8C", "#0B2A5C"]
BULK_COLS = ["#D0662A", "#F0883A", "#FCA95E", "#FDC987", "#FEE5C4", "#FFF3DF"]
POINT_TYPES = [7, 5, 9, 11, 13, 15, 3, 1]


def category_palette(names: list[str]) -> tuple[list[str], list[int]]:
    vac = [n for n in names if n.startswith("Vacancy")]
    bulk = [n for n in names if n.startswith("Bulk")]
    cols = {}
    for i, n in enumerate(vac):
        cols[n] = VAC_COLS[i * len(VAC_COLS) // max(len(vac), 1)] if len(vac) <= len(VAC_COLS) else VAC_COLS[i % len(VAC_COLS)]
    for i, n in enumerate(bulk):
        cols[n] = BULK_COLS[round(i * (len(BULK_COLS) - 1) / max(len(bulk) - 1, 1))]
    cols["Surface"], cols["Strained"] = SURFACE_COL, STRAIN_COL
    # legend order: Strained, Surface, vacancies cold -> hot, bulk cold -> hot (1-based ids)
    key = (["Strained", "Surface"] + vac + bulk[::-1])
    return [cols[n] for n in names], [names.index(k) + 1 for k in key]


def write(path, header, rows, fmt):
    np.savetxt(path, np.asarray(rows, dtype=float), fmt=fmt, header=header)


def run(dataset: str, names: list[str] | None = None, variant: str = "", sampler: str = "fps"):
    cfg = config.load(dataset)
    frames = data.load_frames(dataset)
    cats, cat = data.categories(frames)
    names = list(names or cfg["descriptors"])
    if variant:
        names = [n + variant if n in descriptors.ATOMIC else n for n in names]
    names = [n for n in names if n in descriptors.available(dataset)]
    methods = ["random"] + [selection.method_id(sampler, n) for n in names]
    tag = dataset + variant + ("" if sampler == "fps" else f"_{sampler}")
    out = paths.plot_data_dir(tag)
    A = paths.results_dir(dataset, "analysis")

    # --- meta.gp -------------------------------------------------------------
    cols, key = category_palette(cats)
    gp = lambda arr: ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in arr)
    text = [f"# {dataset}: categories and methods for the gnuplot figures (written by lsp.analysis.export)",
            f'DATASET = "{dataset}"', f'ELEMENT = "{cfg["element"]}"', f'SAMPLER = "{sampler}"', f"NCAT = {len(cats)}",
            f"array CATNAME[NCAT] = [{gp(cats)}]", f"array CATCOL[NCAT] = [{gp(cols)}]",
            f"array CATKEY[NCAT] = [{gp(key)}]", f"NMETH = {len(methods)}", f"NDESC = {len(names)}",
            f"array METH[NMETH] = [{gp(methods)}]",
            f"array METHNAME[NMETH] = [{gp(['Random'] + [descriptors.label(n) for n in names])}]",
            f"array METHCOL[NMETH] = [{gp([descriptors.COLORS['random']] + [descriptors.COLORS.get(descriptors.split_name(n)[0], '#333333') for n in names])}]",
            f"array METHDT[NMETH] = [{gp([2] + [1] * len(names))}]",
            f"array METHPT[NMETH] = [{gp([POINT_TYPES[i % len(POINT_TYPES)] for i in range(len(methods))])}]",
            f"array DESC[NDESC] = [{gp(names)}]",
            f"array DESCNAME[NDESC] = [{gp([descriptors.label(m) for m in names])}]"]
    (out / "meta.gp").write_text("\n".join(text) + "\n")

    # --- embeddings ------------------------------------------------------------
    emb = pd.read_csv(A / "embeddings.csv")
    for n in names:
        e = emb[emb.descriptor == n].sort_values("frame")
        write(out / f"umap_{n}.dat", f"{dataset} {n}: UMAP of the standardized latent space; umap1 umap2 category_id",
              np.column_stack([e.umap1, e.umap2, e.category_id]), ["%.6f", "%.6f", "%d"])
        write(out / f"toc_points_{n}.dat", f"{dataset} {n}: 2D PCA ({e.pca_explained.iloc[0]:.2f} of variance); pc1 pc2 category_id",
              np.column_stack([e.pc1, e.pc2, e.category_id]), ["%.5f", "%.5f", "%d"])
        sel = selection.load(dataset, selection.method_id(sampler, n))
        idx = next(i for p, r, i in selection.iter_cuts(sel) if p == 5 and r == 0)
        write(out / f"toc_selected_{n}.dat", f"{dataset} {n}: PCA coordinates of the FPS picks at 5 % (replicate 1)",
              np.column_stack([e.pc1.to_numpy()[idx], e.pc2.to_numpy()[idx]]), "%.5f")

    # --- composition ---------------------------------------------------------
    comp = pd.read_csv(A / "composition.csv")
    for m in methods:
        c = comp[comp.method == m].sort_values("percent")
        rows = [[p, *np.cumsum(r)] for p, r in zip(c.percent, c[cats].to_numpy())]
        write(out / f"composition_{m}.dat", "retention_pct cumulative composition (%) by category, bottom->top: "
              + " | ".join(cats), rows, "%.4f")

    # --- information (QUESTS) ------------------------------------------------
    qf = A / "quests.csv"
    q = pd.read_csv(qf) if qf.exists() else pd.DataFrame(columns=["method", "percent", "D_rel", "coverage", "H_rel"])
    if not qf.exists():
        print(f"[{tag}] WARNING: {qf} missing — information_*.dat written as 100 %-only placeholders")
    for m in methods:
        g = q[q.method == m].groupby("percent")
        stats = [g[c].agg(["mean", "std"]).fillna(0.0) for c in ("D_rel", "coverage", "H_rel")]
        rows = [[p, *(v for s in stats for v in s.loc[p])] for p in sorted(g.groups)] + [[100, 1, 0, 1, 0, 1, 0]]
        write(out / f"information_{m}.dat", "retention_pct D_rel_mean D_rel_sd coverage_mean coverage_sd H_rel_mean H_rel_sd",
              rows, "%.6f")

    # --- distances -----------------------------------------------------------
    hist = pd.read_csv(A / "distance_histograms.csv")
    dsel = pd.read_csv(A / "distance_selection.csv")
    rho = []
    for n in names:
        h = hist[hist.descriptor == n]
        edges = np.sort(h.bin_left.unique())
        dens = np.array([h[h.category == c].sort_values("bin_left").density.to_numpy() for c in cats])
        cum = np.cumsum(dens, axis=0).T
        cum = np.vstack([cum, cum[-1]])
        write(out / f"distance_hist_{n}.dat", "bin_left_edge cumulative density by category, bottom->top: " + " | ".join(cats),
              np.column_stack([np.append(edges, 1.0), cum]), "%.6f")
        s = dsel[(dsel.descriptor == n) & (dsel.sampler == sampler)].sort_values("category_id")
        rho.append(float(s.spearman_rho.iloc[0]))
        write(out / f"distance_selection_{n}.dat", f"category_id mean_norm_distance within_category_distance {sampler}_enrichment(retention<=10%)",
              np.column_stack([s.category_id, s.mean_distance, s.within_distance, s.enrichment]),
              ["%d", "%.6f", "%.6f", "%.6f"])
    (out / "distance_rho.gp").write_text(
        f"# Spearman rho(mean normalized distance, FPS enrichment) per latent space\n"
        f"array RHO[{len(rho)}] = [{', '.join(f'{r:.2f}' for r in rho)}]   # {', '.join(names)}\n")
    print(f"[{tag}] gnuplot data -> {out}")
