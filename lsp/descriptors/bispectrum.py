"""Bispectrum coefficients (SNAP descriptors) per atom via maml + LAMMPS `compute sna/atom`.

pot_fit=False returns the per-atom bispectrum rows (N x n_B), which are aggregated
by mean + std over atoms like SOAP / Behler-Parrinello. (pot_fit=True would return
SNAP *fitting* rows — one atom-averaged energy row plus 3N force-derivative rows —
whose mean + std injects 1/(3N+1) atom-count factors that made the small surface
cells outliers in the original Si study; fixed 2026-11-03.)

Parameters per dataset ("bispectrum"): rcutfac, twojmax; element_profile r = 0.5
and w = 1 (mlearn convention), so the real-space cutoff is rcutfac x 2 x 0.5 Å.
Needs a LAMMPS executable with the SNAP package (local tool `lmp_serial`).
"""

from __future__ import annotations

import os

import numpy as np

from .. import paths


def _patch_maml_numpy2():
    """maml's dump reader uses the numpy<2 dtype alias 'float_'; map it to float64."""
    import io
    from maml.apps.pes import _lammps
    if getattr(_lammps, "_np2_float_patched", False):
        return

    def _read_dump(file_name, dtype="float64"):
        if dtype == "float_":
            dtype = "float64"
        with open(file_name) as f:
            lines = f.readlines()[9:]
        return np.loadtxt(io.StringIO("".join(lines)), dtype=dtype)

    _lammps._read_dump = _read_dump
    _lammps._np2_float_patched = True


def atomic_descriptors(data, params: dict, element: str) -> list[np.ndarray]:
    lmp = paths.local_tool("lmp_serial")
    os.environ["PATH"] = os.path.dirname(lmp) + os.pathsep + os.environ.get("PATH", "")
    _patch_maml_numpy2()
    from maml.describers import BispectrumCoefficients
    describer = BispectrumCoefficients(
        rcutfac=params["rcutfac"], twojmax=params["twojmax"],
        element_profile={element: {"r": params.get("r", 0.5), "w": params.get("w", 1)}},
        quadratic=False, pot_fit=False, include_stress=False)
    out = []
    for i, d in enumerate(data):
        print(f"  bispectrum {i + 1}/{len(data)}", end="\r", flush=True)
        out.append(np.asarray(describer.transform_one(d["structure"]).values, dtype=float))
    print()
    return out
