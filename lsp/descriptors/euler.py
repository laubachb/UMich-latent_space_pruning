"""Euler characteristic curve (ECC): a topological, structure-level descriptor.

For each frame a Vietoris–Rips filtration is built over the atoms using
minimum-image distances; at every radius r on a fixed grid the Euler
characteristic of the resulting clique complex, truncated at 2-simplices,

    chi(r) = V - E(r) + T(r)

(V atoms, E(r) atom pairs within r, T(r) triangles whose three edges are all
within r) is recorded and divided by V, so the curve is intensive and comparable
across cell sizes. The 64 values of chi(r)/V on 0 <= r <= 6 Å are the frame's
descriptor; no atomic aggregation is involved. Parameters ("euler"): r_max, n_bins.

Note: minimum-image distances count each pair once; in cells shorter than r_max
along a direction the periodic multi-images are not represented (the convention
of the original Si study, kept for continuity).
"""

from __future__ import annotations

import numpy as np

from ..data import to_ase


def ecc(atoms, r_max: float, n_bins: int) -> np.ndarray:
    n = len(atoms)
    dist = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dist, np.inf)
    radii = np.linspace(0.0, r_max, n_bins)
    chi = np.empty(n_bins)
    for k, r in enumerate(radii):
        adj = (dist <= r).astype(np.float64)
        n_edges = adj.sum() / 2.0
        n_tri = np.trace(adj @ adj @ adj) / 6.0
        chi[k] = n - n_edges + n_tri
    return chi / n


def _one(args):
    atoms, r_max, n_bins = args
    return ecc(atoms, r_max, n_bins)


def structure_descriptors(data, params: dict) -> np.ndarray:
    from .aggregate import parallel_frames
    r_max, n_bins = float(params.get("r_max", 6.0)), int(params.get("n_bins", 64))
    return np.array(parallel_frames(_one, [(to_ase(d["structure"]), r_max, n_bins) for d in data], "euler"))
