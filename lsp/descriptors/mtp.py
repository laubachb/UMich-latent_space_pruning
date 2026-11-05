"""Moment tensor potential (MTP) basis functions as a per-atom descriptor.

MTP (Shapeev, Multiscale Model. Simul. 14, 1153 (2016); MLIP-2) writes the energy
of atom i as a linear combination of basis functions B_alpha(n_i), each a full
contraction of moment tensors

    M_{mu,nu}(n_i) = sum_j f_mu(|r_ij|) r_ij (x) ... (x) r_ij     (nu times)

with the sum over neighbours j within the cutoff. The set of contractions is fixed
by the MTP *level*; we read it from the untrained MTP definition files that ship
with MLIP-2 (lsp/mtp_levels/<level>.mtp, BSD-2 licence): `alpha_index_basic`
enumerates the independent components {mu, a, b, c} of the moment tensors
(component x^a y^b z^c of M_{mu, a+b+c}), `alpha_index_times` the pairwise products
that build the contractions, and `alpha_moment_mapping` which of the resulting
scalars are the basis functions.

Radial functions: MLIP learns f_mu as linear combinations of a Chebyshev basis
Q_beta(r) = T_beta(xi) (r_cut - r)^2, xi = (2 r - (r_min + r_cut)) / (r_cut - r_min).
A descriptor must be model-free, so we use the untrained basis directly,
f_mu = Q_mu (radial-coefficient matrix = identity), which gives the linear MTP of
the original formulation. r_min is set per dataset to just below the smallest
interatomic distance in the data (as `mlp train --auto-min-dist` does).

Parameters per dataset ("mtp"): level, cutoff (Å), min_dist (Å, optional; otherwise
derived from the data). The mlearn configs use the levels and cutoffs of the
optimised MTPs of Zuo et al. 2020 (SI Table S10). The per-atom basis-function
vectors (92 / 288 / 864 values at levels 16 / 20 / 24) are aggregated by mean + std
over atoms like the other atom-centred descriptors; the per-atom "mean" is the
MTP energy design row per atom, the analogue of the ChIMES A-matrix descriptor.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from ase.neighborlist import neighbor_list

from .. import paths
from ..data import to_ase


@lru_cache(maxsize=None)
def load_level(level: int) -> dict:
    f = paths.MTP_LEVELS / f"{level:02d}.mtp"
    if not f.exists():
        raise FileNotFoundError(f"no MTP definition for level {level} in {paths.MTP_LEVELS}")
    txt = f.read_text()

    def arr(name):
        m = re.search(rf"{name}\s*=\s*(\{{.*?\}}\s*)\n", txt, re.S)
        body = m.group(1).replace("{", "[").replace("}", "]")
        return np.array(eval(body), dtype=int)  # noqa: S307 (trusted bundled file)

    def scalar(name):
        return int(re.search(rf"{name}\s*=\s*(\d+)", txt).group(1))

    basic = arr("alpha_index_basic")
    times = arr("alpha_index_times")
    mapping = arr("alpha_moment_mapping")
    n_moments = scalar("alpha_moments_count")
    assert basic.shape == (scalar("alpha_index_basic_count"), 4)
    assert times.shape == (scalar("alpha_index_times_count"), 4)
    assert len(mapping) == scalar("alpha_scalar_moments")
    return dict(level=level, basic=basic, times=times, mapping=mapping, n_moments=n_moments,
                n_radial=int(basic[:, 0].max()) + 1, radial_basis_size=scalar("radial_basis_size"))


def chebyshev_radial(r: np.ndarray, n_funcs: int, r_min: float, r_cut: float) -> np.ndarray:
    """Q_mu(r) = T_mu(xi) (r_cut - r)^2 for mu < n_funcs; shape (len(r), n_funcs)."""
    xi = (2.0 * r - (r_min + r_cut)) / (r_cut - r_min)
    T = np.empty((len(r), n_funcs))
    T[:, 0] = 1.0
    if n_funcs > 1:
        T[:, 1] = xi
    for k in range(2, n_funcs):
        T[:, k] = 2.0 * xi * T[:, k - 1] - T[:, k - 2]
    return T * ((r_cut - r) ** 2)[:, None]


def basis_functions(atoms, level: int, r_cut: float, r_min: float) -> np.ndarray:
    """(n_atoms, n_basis) MTP basis functions of every atom of a single-species ase.Atoms."""
    L = load_level(level)
    n = len(atoms)
    i_idx, D, dist = neighbor_list("iDd", atoms, r_cut)
    radial = chebyshev_radial(dist, L["n_radial"], r_min, r_cut)          # (n_pairs, n_radial)
    # monomials x^a y^b z^c of the neighbour vectors for every basic moment
    basic = L["basic"]
    mono = np.ones((len(dist), len(basic)))
    for k, (_, a, b, c) in enumerate(basic):
        if a:
            mono[:, k] *= D[:, 0] ** a
        if b:
            mono[:, k] *= D[:, 1] ** b
        if c:
            mono[:, k] *= D[:, 2] ** c
    terms = radial[:, basic[:, 0]] * mono                                   # (n_pairs, n_basic)
    moments = np.zeros((n, L["n_moments"]))
    for k in range(len(basic)):
        moments[:, k] = np.bincount(i_idx, weights=terms[:, k], minlength=n)
    for i, j, mult, k in L["times"]:
        moments[:, k] += mult * moments[:, i] * moments[:, j]
    return moments[:, L["mapping"]]


def min_distance(data) -> float:
    m = np.inf
    for d in data:
        atoms = to_ase(d["structure"])
        dist = neighbor_list("d", atoms, 6.0)
        if len(dist):
            m = min(m, float(dist.min()))
    return m


def atomic_descriptors(data, params: dict, element: str) -> list[np.ndarray]:
    level, r_cut = int(params["level"]), float(params["cutoff"])
    r_min = params.get("min_dist")
    if r_min is None:
        r_min = 0.99 * min_distance(data)
        print(f"  mtp: r_min set to 0.99 x smallest pair distance = {r_min:.3f} Å")
    from .aggregate import parallel_frames
    return parallel_frames(_one, [(to_ase(d["structure"]), level, r_cut, float(r_min)) for d in data],
                           f"mtp level {level}")


def _one(args):
    atoms, level, r_cut, r_min = args
    return basis_functions(atoms, level, r_cut, r_min)
