#!/usr/bin/env python3
"""
Helpers for unsupervised baseline UMAPs: descriptor load/compute, UMAP, plotting.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from shutil import which

import matplotlib.pyplot as plt
import numpy as np
import umap
from ase import Atoms
from dscribe.descriptors import SOAP
from matplotlib.lines import Line2D
from pymatgen.core import Structure
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
UNSUP_DIR = SCRIPT_DIR.parent
UMAP_DIR = UNSUP_DIR.parent
ROOT_DIR = UMAP_DIR.parents[1]
CACHE_DIR = UMAP_DIR / "descriptor_cache"
CACHE_PATH = CACHE_DIR / "structure_descriptors.npz"
CHIMES_PKL = ROOT_DIR / "descriptors" / "chimes" / "frames_descriptors.pkl"

sys.path.insert(0, str(ROOT_DIR / "analysis"))
from categorization import (  # noqa: E402
    CATEGORIES,
    PHASE_CATEGORIES,
)

UMAP_COMMON = dict(n_components=2, min_dist=0.9, random_state=42)
SOAP_PARAMS = dict(
    species=["Si"],
    periodic=True,
    r_cut=5.1,
    n_max=8,
    l_max=8,
    average="off",
)

CATEGORY_COLORS = {
    "Strained": "#E63946",
    "High Temp (3374K)": "#F77F00",
    "Normal (300K)": "#06A77D",
    "Vacancy (300K)": "#457B9D",
    "Surface": "#A23B72",
    "Other": "#999999",
}

PHASE_CATEGORY_COLORS = {
    "Vacancy-300K": "#457B9D",
    "Vacancy-3374K": "#F77F00",
    "Surface": "#A23B72",
    "Strained Diamond": "#C9A227",
    "Unstrained Diamond": "#06A77D",
    "Liquid": "#E63946",
}


def ensure_lammps_on_path() -> None:
    """Prepend Homebrew LAMMPS bins so maml can find lmp_serial."""
    candidates = [
        "/opt/homebrew/bin",
        "/opt/homebrew/Cellar/lammps/20250722-update4/bin",
        "/opt/homebrew/Cellar/lammps/20250722-update1/bin",
        "/usr/local/bin",
    ]
    path = os.environ.get("PATH", "")
    for p in reversed(candidates):
        if Path(p).is_dir() and p not in path.split(os.pathsep):
            path = p + os.pathsep + path
    os.environ["PATH"] = path


def pymatgen_to_ase(structure: Structure) -> Atoms:
    symbols = [site.species_string for site in structure]
    positions = structure.cart_coords
    cell = structure.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def aggregate_atomic_descriptors(atomic_descriptors, method: str = "mean_std"):
    if method == "mean":
        return np.mean(atomic_descriptors, axis=0)
    if method == "mean_std":
        return np.concatenate(
            [np.mean(atomic_descriptors, axis=0), np.std(atomic_descriptors, axis=0)]
        )
    if method == "sum":
        return np.sum(atomic_descriptors, axis=0)
    raise ValueError(f"Unknown aggregation method: {method}")


def _lammps_on_path() -> bool:
    return any(
        which(name) is not None
        for name in ("lmp_serial", "lmp_mpi", "lmp", "lammps", "lammps_serial")
    )


def compute_soap_descriptors(structures):
    print("Computing SOAP descriptors...")
    soap = SOAP(**SOAP_PARAMS)
    out = []
    for i, structure in enumerate(structures):
        print(f"  SOAP {i + 1}/{len(structures)}", end="\r")
        atomic = soap.create(pymatgen_to_ase(structure))
        out.append(aggregate_atomic_descriptors(atomic, method="mean_std"))
    print()
    return np.array(out)


def compute_behler_descriptors(structures):
    print("Computing Behler descriptors...")
    from maml.describers import BPSymmetryFunctions

    ensure_lammps_on_path()
    describer = BPSymmetryFunctions(
        cutoff=5.5,
        r_etas=[0.01, 0.02, 0.05],
        r_shift=[4.0],
        a_etas=[0.01, 0.02, 0.05],
        zetas=[1.0, 16.0, 2.0, 4.0],
        lambdas=[1, -1],
    )
    out = []
    for i, structure in enumerate(structures):
        print(f"  Behler {i + 1}/{len(structures)}", end="\r")
        atomic = describer.transform_one(structure).values
        out.append(aggregate_atomic_descriptors(atomic, method="mean_std"))
    print()
    return np.array(out)


def compute_bispectrum_descriptors(structures):
    print("Computing Bispectrum descriptors...")
    from maml.describers import BispectrumCoefficients

    ensure_lammps_on_path()
    describer = BispectrumCoefficients(
        rcutfac=4.9,
        twojmax=8,
        element_profile={"Si": {"r": 0.5, "w": 1}},
        quadratic=False,
        pot_fit=True,
        include_stress=False,
    )
    out = []
    for i, structure in enumerate(structures):
        print(f"  Bispectrum {i + 1}/{len(structures)}", end="\r")
        atomic = describer.transform_one(structure).values
        out.append(aggregate_atomic_descriptors(atomic, method="mean_std"))
    print()
    return np.array(out)


def compute_chimes_descriptors(n_structures: int, data: list | None = None):
    """
    Load precomputed ChIMES frame matrices and aggregate mean+std per structure.

    Matches descriptors/chimes/compute_and_prune.py: frame_index defaults to
    list index when absent. Each frame matrix is (3*natoms, n_features).
    """
    import pickle

    if not CHIMES_PKL.is_file():
        raise FileNotFoundError(
            f"Missing {CHIMES_PKL}. Run descriptors/chimes/process_raw_descriptors.py first."
        )
    print(f"Loading ChIMES descriptors from {CHIMES_PKL}...")
    with CHIMES_PKL.open("rb") as f:
        frame_descriptors = pickle.load(f)

    out = []
    for i in range(n_structures):
        frame_index = i
        if data is not None:
            frame_index = data[i].get("frame_index", i)
        if frame_index not in frame_descriptors:
            raise KeyError(f"ChIMES frame_index {frame_index} not in {CHIMES_PKL.name}")
        matrix = np.asarray(frame_descriptors[frame_index], dtype=float)
        if np.isnan(matrix).any():
            matrix = np.nan_to_num(matrix, nan=0.0)
        out.append(aggregate_atomic_descriptors(matrix, method="mean_std"))
    arr = np.array(out)
    arr = np.nan_to_num(arr, nan=0.0)
    print(f"  ChIMES structure descriptors: {arr.shape}")
    return arr


def _save_descriptor_cache(descriptors: dict) -> None:
    payload = {}
    if "Behler Symmetry Functions" in descriptors:
        payload["behler"] = descriptors["Behler Symmetry Functions"]
    if "SOAP" in descriptors:
        payload["soap"] = descriptors["SOAP"]
    if "Bispectrum" in descriptors:
        payload["bispectrum"] = descriptors["Bispectrum"]
    if "ChIMES" in descriptors:
        payload["chimes"] = descriptors["ChIMES"]
    if payload:
        np.savez(CACHE_PATH, **payload)
        print(f"Cached descriptors ({', '.join(payload)}) to {CACHE_PATH}")


def _resolve_cache_path() -> Path:
    """Prefer descriptor_cache/; fall back to legacy .cache/ name."""
    if CACHE_PATH.exists():
        return CACHE_PATH
    legacy = UMAP_DIR / ".cache" / "descriptors_cache.npz"
    if legacy.exists():
        print(f"Note: using legacy cache path {legacy}")
        print(f"      (preferred location is {CACHE_PATH})")
        return legacy
    return CACHE_PATH


def load_or_compute_descriptors(
    structures,
    force_recompute: bool = False,
    data: list | None = None,
):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    descriptors: dict = {}
    cache_path = _resolve_cache_path()

    if cache_path.exists() and not force_recompute:
        print(f"Loading cached descriptors from {cache_path}")
        cache = np.load(cache_path)
        if "behler" in cache.files:
            descriptors["Behler Symmetry Functions"] = cache["behler"]
        if "soap" in cache.files:
            descriptors["SOAP"] = cache["soap"]
        if "bispectrum" in cache.files:
            descriptors["Bispectrum"] = cache["bispectrum"]
        if "chimes" in cache.files:
            descriptors["ChIMES"] = cache["chimes"]

    ensure_lammps_on_path()

    if "SOAP" not in descriptors:
        descriptors["SOAP"] = compute_soap_descriptors(structures)
        _save_descriptor_cache(descriptors)

    if "ChIMES" not in descriptors:
        try:
            descriptors["ChIMES"] = compute_chimes_descriptors(len(structures), data=data)
            _save_descriptor_cache(descriptors)
        except Exception as exc:
            print(f"WARNING: ChIMES failed ({exc})")

    if not _lammps_on_path():
        print(
            "WARNING: LAMMPS executable not on PATH — skipping Behler/Bispectrum. "
            "Need lmp_serial / lmp_mpi (Homebrew) or lmp."
        )
        return descriptors

    if "Behler Symmetry Functions" not in descriptors:
        try:
            descriptors["Behler Symmetry Functions"] = compute_behler_descriptors(structures)
            _save_descriptor_cache(descriptors)
        except Exception as exc:
            print(f"WARNING: Behler failed ({exc})")

    if "Bispectrum" not in descriptors:
        try:
            descriptors["Bispectrum"] = compute_bispectrum_descriptors(structures)
            _save_descriptor_cache(descriptors)
        except Exception as exc:
            print(f"WARNING: Bispectrum failed ({exc})")

    return descriptors


def compute_umap_embedding(
    descriptors: np.ndarray,
    init: str = "spectral",
    min_dist: float | None = None,
    metric: str | None = None,
    n_neighbors: int | None = None,
    n_components: int | None = None,
    y: np.ndarray | None = None,
) -> np.ndarray:
    scaled = StandardScaler().fit_transform(descriptors)
    params = {**UMAP_COMMON, "init": init}
    if min_dist is not None:
        params["min_dist"] = min_dist
    if metric is not None:
        params["metric"] = metric
    if n_neighbors is not None:
        params["n_neighbors"] = n_neighbors
    if n_components is not None:
        params["n_components"] = n_components
    reducer = umap.UMAP(**params)
    if y is None:
        return reducer.fit_transform(scaled)
    return reducer.fit_transform(scaled, y=y)


def _panel_order(descriptor_dict: dict) -> list[str]:
    preferred = ["Behler Symmetry Functions", "SOAP", "Bispectrum", "ChIMES"]
    return [k for k in preferred if k in descriptor_dict] or list(descriptor_dict)


def plot_descriptor_row(
    axes,
    embeddings: dict[str, np.ndarray],
    colors: np.ndarray,
    colorbar_label: str,
    titles: dict[str, str] | None = None,
    shared_colorbar: bool = True,
    cax=None,
):
    names = _panel_order(embeddings)
    sc = None
    for ax, name in zip(axes, names):
        sc = ax.scatter(
            embeddings[name][:, 0],
            embeddings[name][:, 1],
            c=colors,
            s=12,
            alpha=0.75,
            cmap="viridis",
        )
        ax.set_title(titles[name] if titles else name, fontsize=13)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    if sc is None:
        return
    if cax is not None:
        axes[0].figure.colorbar(sc, cax=cax, label=colorbar_label)
    elif shared_colorbar:
        fig = axes[0].figure
        fig.colorbar(sc, ax=list(axes), label=colorbar_label, fraction=0.02, pad=0.02)


def plot_category_row(
    axes,
    embeddings: dict[str, np.ndarray],
    categories: list[str],
    titles: dict[str, str] | None = None,
):
    names = _panel_order(embeddings)
    cats = np.asarray(categories)
    for ax, name in zip(axes, names):
        for cat in CATEGORIES:
            mask = cats == cat
            if not np.any(mask):
                continue
            ax.scatter(
                embeddings[name][mask, 0],
                embeddings[name][mask, 1],
                c=CATEGORY_COLORS[cat],
                s=18,
                alpha=0.8,
                label=cat,
                edgecolors="none",
            )
        ax.set_title(titles[name] if titles else name, fontsize=13)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CATEGORY_COLORS[cat],
            markersize=8,
            label=cat,
        )
        for cat in CATEGORIES
        if cat in cats
    ]
    axes[-1].legend(handles=handles, loc="best", fontsize=9, frameon=True, title="Category")


def plot_phase_category_row(
    axes,
    embeddings: dict[str, np.ndarray],
    phase_categories: list[str],
    titles: dict[str, str] | None = None,
):
    """Color by full phase categories."""
    names = _panel_order(embeddings)
    cats = np.asarray(phase_categories)
    for ax, name in zip(axes, names):
        for cat in PHASE_CATEGORIES:
            mask = cats == cat
            if not np.any(mask):
                continue
            ax.scatter(
                embeddings[name][mask, 0],
                embeddings[name][mask, 1],
                c=PHASE_CATEGORY_COLORS[cat],
                s=18,
                alpha=0.8,
                label=cat,
                edgecolors="none",
            )
        ax.set_title(titles[name] if titles else name, fontsize=13)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PHASE_CATEGORY_COLORS[cat],
            markersize=8,
            label=cat,
        )
        for cat in PHASE_CATEGORIES
        if cat in cats
    ]
    axes[-1].legend(
        handles=handles, loc="best", fontsize=9, frameon=True, title="Phase category"
    )
