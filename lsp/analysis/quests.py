"""QUESTS information content of the selected training sets.

Every cut is scored in ONE shared, descriptor-agnostic atomic space — the QUESTS
representation (Schwalbe-Koda et al., Nat. Commun. 2025; k = 32 neighbours,
5 Å cutoff, Gaussian kernel bandwidth h = 0.015) — so the latent space used to
SELECT the frames is never the space used to JUDGE them. With x_i the QUESTS
descriptors of all atoms of a cut S and p_i(S) = sum_{j in S} K(x_i, x_j):

    D(S) = log sum_{i in S} 1/p_i(S)             diversity: log of the effective number
                                                 of distinct atomic environments
    H(S) = -1/|S| sum_{i in S} log(p_i(S)/|S|)   entropy: average surprisal (uniformity)
    coverage(S) = mean over ALL atoms y of the full dataset of [ p_y(S) >= 1 ]
                  (i.e. dH(y|S) = -log p_y(S) <= 0: y is not novel w.r.t. S)

reported relative to the full dataset (D/D_full, H/H_full; coverage is 1 at 100 %).

Implementation. A cut is a set of whole frames and the kernel sum is additive over
reference atoms, so p_i(S) = sum_{f in S} B[i, f] with the frame-blocked kernel matrix

    B[i, f] = sum_{j in frame f} K(x_i, x_j)        (n_atoms x n_frames, computed once)

Every cut is then one column-subset sum of B — exact, and thousands of times cheaper
than re-evaluating the kernel per cut, which is what makes 100 replicates practical.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import data, paths, selection

BATCH = 256
CHUNK = 256          # cuts scored per matrix product


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


def frame_kernel(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """(B, n_atoms): B[i, f] = sum over atoms j of frame f of K(x_i, x_j), in float64."""
    from quests.entropy import DEFAULT_BANDWIDTH, kernel_sum
    x, n_atoms = atomic_descriptors(dataset)
    cache = paths.results_dir(dataset, "descriptors") / "quests_frame_kernel.npy"
    if cache.exists():
        B = np.load(cache)
        if B.shape == (x.shape[0], len(n_atoms)):
            return B, n_atoms
    t0 = time.time()
    x = np.ascontiguousarray(x, dtype=np.float64)
    off = np.concatenate([[0], np.cumsum(n_atoms)])
    B = np.empty((x.shape[0], len(n_atoms)))
    for f in range(len(n_atoms)):
        B[:, f] = kernel_sum(x, x[off[f]:off[f + 1]], h=DEFAULT_BANDWIDTH, batch_size=BATCH)
    np.save(cache, B)
    print(f"[{dataset}] QUESTS frame kernel {B.shape} in {time.time() - t0:.0f} s")
    return B, n_atoms


def score_cuts(B: np.ndarray, n_atoms: np.ndarray, cuts: list[np.ndarray]) -> np.ndarray:
    """(n_cuts, 4) array of H, D, coverage, n_atoms_in_cut for frame-index cuts."""
    frame_of_atom = np.repeat(np.arange(len(n_atoms)), n_atoms)
    out = np.empty((len(cuts), 4))
    for c0 in range(0, len(cuts), CHUNK):
        chunk = cuts[c0:c0 + CHUNK]
        ind = np.zeros((len(n_atoms), len(chunk)))
        for k, idx in enumerate(chunk):
            ind[idx, k] = 1.0
        P = B @ ind                                   # p_i(S) for every atom i and cut S
        inside = ind[frame_of_atom] > 0               # atom i belongs to cut S
        nS = inside.sum(axis=0)
        with np.errstate(divide="ignore"):
            logp = np.log(P)
        out[c0:c0 + len(chunk), 0] = -(np.where(inside, logp, 0.0).sum(axis=0) / nS - np.log(nS))
        out[c0:c0 + len(chunk), 1] = np.log(np.where(inside, 1.0 / P, 0.0).sum(axis=0))
        out[c0:c0 + len(chunk), 2] = (P >= 1.0).mean(axis=0)
        out[c0:c0 + len(chunk), 3] = nS
    return out


def run(dataset: str, methods: list[str] | None = None) -> pd.DataFrame:
    B, n_atoms = frame_kernel(dataset)
    full = score_cuts(B, n_atoms, [np.arange(len(n_atoms))])[0]
    h_full, d_full = full[0], full[1]
    print(f"[{dataset}] H(full) = {h_full:.4f}, D(full) = {d_full:.4f} nats over {B.shape[0]} atoms")
    methods = methods or selection.available(dataset)
    t0 = time.time()
    rows = []
    for m in methods:
        sel = selection.load(dataset, m)
        meta = [(p, r, idx) for p, r, idx in selection.iter_cuts(sel)]
        s = score_cuts(B, n_atoms, [idx for _, _, idx in meta])
        for (p, r, idx), (hh, dd, cov, na) in zip(meta, s):
            rows.append(dict(method=m, percent=p, replicate=r + 1, n_frames=len(idx), n_atoms=int(na),
                             H=hh, D=dd, H_rel=hh / h_full, D_rel=dd / d_full, coverage=cov,
                             H_full=h_full, D_full=d_full))
    df = pd.DataFrame(rows).sort_values(["method", "percent", "replicate"])
    out = paths.results_dir(dataset, "analysis") / "quests.csv"
    tmp = out.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"[{dataset}] quests: {len(df)} cuts of {len(methods)} methods in {time.time() - t0:.0f} s -> {out}")
    return df
