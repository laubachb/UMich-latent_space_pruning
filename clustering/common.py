"""
Shared utilities for descriptor-space clustering analysis.

Provides:
  * load_frames()          — load data.json (pymatgen structures + raw entries)
  * extract_metadata()     — per-frame physical/categorical metadata for coloring
  * DESCRIPTOR_BUILDERS    — registry of structure-level featurizers, one per
                             descriptor space. Each builder lazily imports its
                             heavy dependencies, so a missing library (dscribe,
                             maml, LAMMPS, a ChIMES pkl, ...) only skips that one
                             descriptor instead of breaking the whole analysis.
  * compute_descriptor_matrix() / cache helpers

Every featurizer returns a RAW (unstandardized) (n_frames, n_features) matrix.
Standardization is applied downstream at analysis time so the cache stays
descriptor-agnostic.
"""

from __future__ import annotations

import os

# Some HPC nodes deadlock in the OpenMP/BLAS thread pools used by sklearn's
# t-SNE / KMeans (the process hangs indefinitely). Pin the thread counts to 1
# BEFORE numpy/BLAS are imported to keep the analysis reliable; these datasets
# are tiny (a few hundred frames) so single-threaded is plenty fast. Override by
# exporting OMP_NUM_THREADS etc. in the environment if your node is unaffected.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re
import pickle
from pathlib import Path

import numpy as np
from monty.serialization import loadfn

# Directory layout ------------------------------------------------------------
CLUSTERING_DIR = Path(__file__).parent
ROOT_DIR = CLUSTERING_DIR.parent
DATA_JSON = ROOT_DIR / "data.json"
CACHE_DIR = CLUSTERING_DIR / "cache"
FIGURES_DIR = ROOT_DIR / "figures"

CACHE_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ── Data + metadata ─────────────────────────────────────────────────────────
def load_frames():
    """Return the deserialized list of data.json entries (pymatgen-backed)."""
    if not DATA_JSON.exists():
        raise FileNotFoundError(
            f"{DATA_JSON} not found. Copy data.json to the project root first."
        )
    data = loadfn(str(DATA_JSON))
    print(f"Loaded {len(data)} frames from {DATA_JSON}")
    return data


def _categorize(description: str) -> str:
    """Coarse physical category (mirrors analysis/normalized_category_composition.py)."""
    desc = description.lower()
    if "strain" in desc or "mode" in desc:
        return "Strained"
    elif "vacancy" in desc:
        return "High Temp Vacancy" if "3374 K" in description else "Vacancy"
    elif "surface" in desc:
        return "Surface"
    elif "3374 K" in description:
        return "High Temp"
    elif "300 K" in description:
        return "Normal (300K)"
    return "Other"


def _parse_temperature(description: str):
    m = re.search(r"(\d+)\s*K", description)
    return float(m.group(1)) if m else np.nan


def _parse_strain(description: str):
    m = re.search(r"strain\s+(-?\d+\.?\d*)", description.lower())
    return float(m.group(1)) if m else np.nan


def extract_metadata(data) -> dict:
    """Per-frame metadata used to color / label the latent spaces.

    Returns a dict of equal-length lists/arrays. Mix of categorical (group,
    category) and continuous physical (temperature, energy/atom, mean normal
    stress) handles so we can test whether latent-space geometry tracks physics.
    """
    groups, categories, temps, strains = [], [], [], []
    energy_per_atom, mean_normal_stress, num_atoms = [], [], []

    for d in data:
        desc = d.get("description", "")
        n = d.get("num_atoms") or len(d["structure"])
        out = d.get("outputs", {}) or {}

        groups.append(d.get("group", "Unknown"))
        categories.append(_categorize(desc))
        temps.append(_parse_temperature(desc))
        strains.append(_parse_strain(desc))
        num_atoms.append(n)

        energy = out.get("energy", np.nan)
        energy_per_atom.append(energy / n if energy is not None and n else np.nan)

        vs = out.get("virial_stress", None)
        if vs is not None and len(vs) >= 3:
            mean_normal_stress.append(float(np.mean(np.asarray(vs, float)[:3])))
        else:
            mean_normal_stress.append(np.nan)

    return {
        "group": np.array(groups),
        "category": np.array(categories),
        "temperature_K": np.array(temps, dtype=float),
        "strain": np.array(strains, dtype=float),
        "energy_per_atom": np.array(energy_per_atom, dtype=float),
        "mean_normal_stress": np.array(mean_normal_stress, dtype=float),
        "num_atoms": np.array(num_atoms, dtype=float),
    }


# ── Structure-level featurizers (one per descriptor space) ───────────────────
def _pymatgen_to_ase(structure):
    from ase import Atoms
    return Atoms(
        symbols=[site.species_string for site in structure],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True,
    )


def _mean_std(atomic_descriptors):
    a = np.asarray(atomic_descriptors, dtype=float)
    return np.concatenate([a.mean(axis=0), a.std(axis=0)])


def build_euler(data):
    """Euler characteristic curve (topological). numpy + ase only.

    Mirrors descriptors/euler/compute_and_prune.py so the clustering package is
    self-contained (that script imports skmatter at module scope, so it cannot be
    imported here without the FPS dependency)."""
    R_MAX, N_BINS = 6.0, 64
    radii = np.linspace(0.0, R_MAX, N_BINS)

    feats = []
    for i, d in enumerate(data):
        print(f"  euler {i + 1}/{len(data)}", end="\r")
        atoms = _pymatgen_to_ase(d["structure"])
        n = len(atoms)
        dist = atoms.get_all_distances(mic=True)
        np.fill_diagonal(dist, np.inf)
        chi = np.empty(len(radii))
        for k, r in enumerate(radii):
            adj = (dist <= r).astype(np.float64)
            n_edges = adj.sum() / 2.0
            n_triangles = np.trace(adj @ adj @ adj) / 6.0
            chi[k] = n - n_edges + n_triangles
        feats.append(chi / n)  # per-atom (intensive), comparable across cell sizes
    print()
    return np.array(feats)


def build_soap(data):
    from dscribe.descriptors import SOAP
    soap = SOAP(species=["Si"], r_cut=5.0, n_max=9, l_max=9, periodic=True)
    feats = []
    for i, d in enumerate(data):
        print(f"  soap {i + 1}/{len(data)}", end="\r")
        atoms = _pymatgen_to_ase(d["structure"])
        feats.append(_mean_std(soap.create(atoms)))
    print()
    return np.array(feats)


def _patch_maml_numpy2():
    """Make maml's LAMMPS dump reader work under numpy>=2.

    maml.apps.pes._lammps._read_dump defaults to the dtype string 'float_',
    an alias numpy removed in 2.0, so np.loadtxt raises
    "data type 'float_' not understood". All internal callers reference the
    function via the module global, so replacing it here fixes every call
    (notably Bispectrum's dump.sna / dump.snad / dump.snav reads). Idempotent.
    """
    import io
    from maml.apps.pes import _lammps
    if getattr(_lammps, "_np2_float_patched", False):
        return

    def _read_dump(file_name, dtype="float64"):
        if dtype == "float_":  # numpy<2 alias -> numpy>=2 canonical name
            dtype = "float64"
        with open(file_name) as f:
            lines = f.readlines()[9:]
        return np.loadtxt(io.StringIO("".join(lines)), dtype=dtype)

    _lammps._read_dump = _read_dump
    _lammps._np2_float_patched = True


def build_behler(data):
    import os
    _patch_maml_numpy2()
    from maml.describers import BPSymmetryFunctions
    os.environ.setdefault("PATH", "")  # ensure lmp is discoverable on user's PATH
    describer = BPSymmetryFunctions(
        cutoff=5.5,
        r_etas=[0.01, 0.02, 0.05],
        r_shift=[4.0],
        a_etas=[0.01, 0.02, 0.05],
        zetas=[1.0, 16.0, 2.0, 4.0],
        lambdas=[1, -1],
    )
    feats = []
    for i, d in enumerate(data):
        print(f"  behler {i + 1}/{len(data)}", end="\r")
        feats.append(_mean_std(describer.transform_one(d["structure"]).values))
    print()
    return np.array(feats)


def build_bispectrum(data):
    _patch_maml_numpy2()
    from maml.describers import BispectrumCoefficients
    describer = BispectrumCoefficients(
        rcutfac=4.9,
        twojmax=8,
        element_profile={"Si": {"r": 0.5, "w": 1}},
        quadratic=False,
        pot_fit=True,
        include_stress=False,
    )
    feats = []
    for i, d in enumerate(data):
        print(f"  bispectrum {i + 1}/{len(data)}", end="\r")
        feats.append(_mean_std(describer.transform_one(d["structure"]).values))
    print()
    return np.array(feats)


def build_chimes(data):
    pkl = ROOT_DIR / "descriptors" / "chimes" / "frames_descriptors.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"{pkl} not found (run chimes process_raw_descriptors.py)")
    with open(pkl, "rb") as f:
        frame_descriptors = pickle.load(f)
    feats = []
    for i, d in enumerate(data):
        frame_index = d.get("frame_index", i)
        if frame_index is None:
            frame_index = i
        if frame_index not in frame_descriptors:
            raise KeyError(f"ChIMES frame index {frame_index} missing; cannot align descriptors")
        feats.append(_mean_std(frame_descriptors[frame_index]))
    return np.array(feats)


DESCRIPTOR_BUILDERS = {
    "euler": build_euler,
    "soap": build_soap,
    "behler": build_behler,
    "bispectrum": build_bispectrum,
    "chimes": build_chimes,
}

# Human-readable display names for figures (acronyms/casing that .capitalize()
# would get wrong, e.g. SOAP, ChIMES).
DESCRIPTOR_LABELS = {
    "euler": "Euler",
    "soap": "SOAP",
    "behler": "Behler",
    "bispectrum": "Bispectrum",
    "chimes": "ChIMES",
}


def descriptor_label(name: str) -> str:
    """Display name for a descriptor, falling back to a capitalized key."""
    return DESCRIPTOR_LABELS.get(name, name.capitalize())


# Consistent colors for descriptors across all figures.
DESCRIPTOR_COLORS = {
    "euler": "#8338EC",
    "soap": "#3A86FF",
    "behler": "#06A77D",
    "bispectrum": "#F77F00",
    "chimes": "#E63946",
}


# ── Cache helpers ────────────────────────────────────────────────────────────
def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}_descriptors.npy"


def save_descriptor_matrix(name: str, matrix: np.ndarray):
    np.save(cache_path(name), matrix)
    print(f"  cached {name}: {matrix.shape} -> {cache_path(name)}")


def load_descriptor_matrix(name: str):
    p = cache_path(name)
    return np.load(p) if p.exists() else None


def available_descriptors():
    """Descriptor names that have a cached matrix on disk (in registry order)."""
    return [n for n in DESCRIPTOR_BUILDERS if cache_path(n).exists()]


def load_all_descriptor_matrices():
    """Return {name: raw_matrix} for every cached descriptor."""
    return {n: load_descriptor_matrix(n) for n in available_descriptors()}


def standardize(matrix: np.ndarray) -> np.ndarray:
    """Zero-mean/unit-variance per feature; zero-variance columns collapse to 0."""
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(matrix)
