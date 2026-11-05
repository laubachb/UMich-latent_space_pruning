"""SOAP power spectrum (dscribe), per atom, aggregated by mean + std over atoms.

Parameters per dataset (configs/<ds>.json, "soap"): r_cut, n_max, l_max, sigma.
The mlearn configs use the SOAP settings of the GAP models of Zuo et al. 2020
(gap.xml: n_max = l_max = 8, atom_sigma = 0.5 Å, per-element cutoff).
"""

from __future__ import annotations

import numpy as np

from ..data import to_ase


def atomic_descriptors(data, params: dict, element: str) -> list[np.ndarray]:
    from dscribe.descriptors import SOAP
    soap = SOAP(species=[element], r_cut=params["r_cut"], n_max=params["n_max"],
                l_max=params["l_max"], sigma=params.get("sigma", 1.0), periodic=True)
    from .aggregate import parallel_frames
    return parallel_frames(_one, [(soap, to_ase(d["structure"])) for d in data], "soap")


def _one(args):
    soap, atoms = args
    return np.asarray(soap.create(atoms), dtype=float)
