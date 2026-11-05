"""Behler-Parrinello atom-centred symmetry functions (radial G2 + angular G4).

    G2_i = sum_j exp(-eta (r_ij - r_s)^2) fc(r_ij)
    G4_i = 2^(1-zeta) sum_{j<k} (1 + lambda cos theta_ijk)^zeta
           exp(-eta' (r_ij^2 + r_ik^2 + r_jk^2)) fc(r_ij) fc(r_ik) fc(r_jk)
    fc(r) = 0.5 (cos(pi r / r_c) + 1)        (n2p2 cutoff_type 1)

The function set is the product grid r_etas x r_shift (G2) and
a_etas x zetas x lambdas (G4), in that column order — the enumeration of
maml.describers.BPSymmetryFunctions, which the original Si study used
(tests/test_legacy_si.py checks the two agree to machine precision).

Units: all lengths in Å and all etas in Å^-2. NOTE for the mlearn datasets: the
NNPs of Zuo et al. were trained with RuNNer/n2p2, which take Bohr units; mlearn
converted r_c and r_s to Bohr but wrote the eta values unchanged, so the published
models (and SI Tables S1-S6, which list eta = 0.036, 0.071, 0.179 ... Å^-2) use
eta_[Å^-2] = eta_nominal / 0.529177^2. The mlearn configs store the converted
Å^-2 values; si_legacy keeps the nominal numbers used by the original Si study
(0.01, 0.02, 0.05 Å^-2, i.e. 3.6x broader Gaussians than the paper's NNP).

Per-atom vectors are aggregated by mean + std over atoms.
"""

from __future__ import annotations

import itertools

import numpy as np
from ase.neighborlist import neighbor_list

from ..data import to_ase


def _fc(r, rc):
    return np.where(r < rc, 0.5 * (np.cos(np.pi * r / rc) + 1.0), 0.0)


def symmetry_functions(atoms, cutoff, r_etas, r_shift, a_etas, zetas, lambdas):
    """(n_atoms, n_G2 + n_G4) symmetry functions of a single-species ase.Atoms."""
    n = len(atoms)
    i_idx, j_idx, D, dist = neighbor_list("ijDd", atoms, cutoff)
    g2_grid = list(itertools.product(r_etas, r_shift))
    g4_grid = list(itertools.product(a_etas, zetas, lambdas))      # maml column order
    out = np.zeros((n, len(g2_grid) + len(g4_grid)))

    fc = _fc(dist, cutoff)
    for k, (eta, rs) in enumerate(g2_grid):
        np.add.at(out[:, k], i_idx, np.exp(-eta * (dist - rs) ** 2) * fc)

    # angular part: all unordered neighbour pairs (j, k) of each centre i
    order = np.argsort(i_idx, kind="stable")
    i_sorted, D, dist, fc = i_idx[order], D[order], dist[order], fc[order]
    starts = np.searchsorted(i_sorted, np.arange(n + 1))
    a_etas, zetas, lambdas = map(np.asarray, (a_etas, zetas, lambdas))
    for i in range(n):
        s, e = starts[i], starts[i + 1]
        m = e - s
        if m < 2:
            continue
        jj, kk = np.triu_indices(m, k=1)
        v1, v2 = D[s + jj], D[s + kk]
        d1, d2 = dist[s + jj], dist[s + kk]
        d3 = np.linalg.norm(v1 - v2, axis=1)
        cos = np.sum(v1 * v2, axis=1) / (d1 * d2)
        cut = fc[s + jj] * fc[s + kk] * _fc(d3, cutoff)
        r2 = d1 ** 2 + d2 ** 2 + d3 ** 2
        vals = []
        for eta, zeta, lam in g4_grid:
            vals.append(np.sum((1.0 + lam * cos) ** zeta * np.exp(-eta * r2) * cut) * 2.0 ** (1.0 - zeta))
        out[i, len(g2_grid):] = vals
    return out


def _one(args):
    atoms, p = args
    return symmetry_functions(atoms, p["cutoff"], p["r_etas"], p["r_shift"], p["a_etas"], p["zetas"], p["lambdas"])


def atomic_descriptors(data, params: dict, element: str) -> list[np.ndarray]:
    from .aggregate import parallel_frames
    return parallel_frames(_one, [(to_ase(d["structure"]), params) for d in data], "behler")
