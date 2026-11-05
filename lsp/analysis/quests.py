"""QUESTS information content of the selected training sets.

Every cut is scored in ONE shared, descriptor-agnostic atomic space — the QUESTS
representation (Schwalbe-Koda et al., Nat. Commun. 2025; k = 32 neighbours,
5 Å cutoff, Gaussian kernel bandwidth h = 0.015) — so the latent space used to
SELECT the frames is never the space used to JUDGE them. With x_i the QUESTS
descriptors of all atoms of a cut S and p_i = sum_j K(x_i, x_j):

    D(S) = log sum_i 1/p_i          diversity: log of the effective number of
                                    distinct atomic environments (subset-monotone)
    H(S) = -1/|S| sum_i log(p_i/|S|) entropy: average surprisal (a uniformity measure)
    coverage(S) = mean over atoms y of the FULL dataset of [ -log sum_{j in S} K(y, x_j) <= 0 ]

reported relative to the full dataset (D/D_full, H/H_full; coverage is 1 at 100 %).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import data, paths, selection

BATCH = 256


def atomic_descriptors(dataset: str):
    from quests.descriptor import DEFAULT_CUTOFF, DEFAULT_K, get_descriptors
    frames = data.load_frames(dataset)
    n_atoms = np.array([len(d["structure"]) for d in frames])
    cache = paths.results_dir(dataset, "descriptors") / "quests_atomic.npy"
    if cache.exists():
        x = np.load(cache)
        if x.shape[0] == n_atoms.sum():
            return x, n_atoms
    t0 = time.time()
    x = get_descriptors([data.to_ase(d["structure"]) for d in frames], k=DEFAULT_K, cutoff=DEFAULT_CUTOFF)
    np.save(cache, x)
    print(f"[{dataset}] QUESTS descriptors {x.shape} in {time.time() - t0:.0f} s")
    return x, n_atoms


def entropy_diversity(x, h):
    from quests.entropy import kernel_sum
    p = kernel_sum(x, x, h=h, batch_size=BATCH)
    return float(-np.mean(np.log(p / x.shape[0]))), float(np.log(np.sum(1.0 / p)))


def coverage(x_full, x_sub, h):
    from quests.entropy import kernel_sum
    p = kernel_sum(x_full, x_sub, h=h, batch_size=BATCH)
    return float(np.mean(-np.log(p) <= 0.0))


def run(dataset: str, methods: list[str] | None = None) -> pd.DataFrame:
    from quests.entropy import DEFAULT_BANDWIDTH
    h = DEFAULT_BANDWIDTH
    x, n_atoms = atomic_descriptors(dataset)
    off = np.concatenate([[0], np.cumsum(n_atoms)])
    rows_of = [np.arange(off[i], off[i + 1]) for i in range(len(n_atoms))]
    h_full, d_full = entropy_diversity(x, h)
    print(f"[{dataset}] H(full) = {h_full:.4f}, D(full) = {d_full:.4f} nats over {x.shape[0]} atoms")
    methods = methods or selection.available(dataset)
    out = paths.results_dir(dataset, "analysis") / "quests.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    rows = [] if done.empty else done.to_dict("records")
    have = {(r["method"], r["percent"], r["replicate"]) for r in rows}
    t0 = time.time()
    for m in methods:
        sel = selection.load(dataset, m)
        for p, r, idx in selection.iter_cuts(sel):
            if (m, p, r + 1) in have:
                continue
            sub = x[np.concatenate([rows_of[i] for i in idx])]
            hh, dd = entropy_diversity(sub, h)
            rows.append(dict(method=m, percent=p, replicate=r + 1, n_frames=len(idx), n_atoms=sub.shape[0],
                             H=hh, D=dd, H_rel=hh / h_full, D_rel=dd / d_full, coverage=coverage(x, sub, h),
                             H_full=h_full, D_full=d_full))
        pd.DataFrame(rows).to_csv(out, index=False)          # checkpoint after every method
        print(f"[{dataset}] quests '{m}' done ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["method", "percent", "replicate"])
    df.to_csv(out, index=False)
    return df
