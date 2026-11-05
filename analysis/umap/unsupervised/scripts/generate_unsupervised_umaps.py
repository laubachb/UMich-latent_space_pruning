#!/usr/bin/env python3
"""
Regenerate unsupervised baseline UMAP figures (category + energy overlays).

Includes Behler, SOAP, Bispectrum, and ChIMES. No figure-level titles
(panel labels identify descriptors).

Writes under analysis/umap/unsupervised/figures/:
  original_categories/umap_by_category.png
  phase_categories/umap_by_phase.png
  energy_per_atom/umap_by_energy.png

Run from repo root:
    python analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure

SCRIPT_DIR = Path(__file__).resolve().parent
UNSUP_DIR = SCRIPT_DIR.parent
UMAP_DIR = UNSUP_DIR.parent
ROOT_DIR = UMAP_DIR.parents[1]
FIG_DIR = UNSUP_DIR / "figures"
PHASE_DATA = UNSUP_DIR / "data" / "data_phase_categories.json"

sys.path.insert(0, str(ROOT_DIR / "analysis"))
sys.path.insert(0, str(SCRIPT_DIR))

from categorization import (  # noqa: E402
    PHASE_CATEGORIES,
    categorize_phase,
    categorize_structure,
)
import umap_helpers as h  # noqa: E402

ORIG_CAT = FIG_DIR / "original_categories" / "umap_by_category.png"
ORIG_E = FIG_DIR / "energy_per_atom" / "umap_by_energy.png"
PHASE_CAT = FIG_DIR / "phase_categories" / "umap_by_phase.png"


def _save_energy_figure(path: Path, embeddings, energy_per_atom, titles) -> None:
    names = h._panel_order(embeddings)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n + 1.0, 4.8), squeeze=False)
    h.plot_descriptor_row(
        axes[0],
        embeddings,
        energy_per_atom,
        "Energy / atom (eV)",
        titles=titles,
        shared_colorbar=False,
    )
    fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.12, wspace=0.22)
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.75])
    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(
            vmin=float(energy_per_atom.min()), vmax=float(energy_per_atom.max())
        ),
    )
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="Energy / atom (eV)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    h.ensure_lammps_on_path()
    data_path = ROOT_DIR / "data.json"
    if not data_path.is_file():
        sys.exit(f"Missing {data_path}")

    with data_path.open() as f:
        data = json.load(f)

    if PHASE_DATA.is_file():
        with PHASE_DATA.open() as f:
            phase_data = json.load(f)
        if len(phase_data) == len(data):
            data = phase_data

    structures = [Structure.from_dict(d["structure"]) for d in data]
    descriptions = [d.get("description") or "" for d in data]
    original_cats = [
        d.get("category") or categorize_structure(desc)
        for d, desc in zip(data, descriptions)
    ]
    phase_cats = [
        d.get("phase_category")
        or categorize_phase(d.get("description") or "", d.get("group"))
        for d in data
    ]
    n_atoms = np.array(
        [d.get("num_atoms", len(s)) for d, s in zip(data, structures)], dtype=float
    )
    energy_total = np.array([d["outputs"]["energy"] for d in data], dtype=float)
    energy_per_atom = energy_total / n_atoms

    force = "--force-recompute" in sys.argv
    descriptors = h.load_or_compute_descriptors(
        structures, force_recompute=force, data=data
    )
    if not descriptors:
        sys.exit("No descriptors available.")
    names = h._panel_order(descriptors)
    print("Descriptors:", ", ".join(names))
    if "ChIMES" not in descriptors:
        sys.exit(
            "ChIMES descriptor missing — check descriptors/chimes/frames_descriptors.pkl"
        )

    print("Computing unsupervised baseline UMAP embeddings...")
    embeddings = {
        name: h.compute_umap_embedding(desc, init="spectral")
        for name, desc in descriptors.items()
    }
    n = len(names)
    titles = {k: k for k in names}

    print("Writing original-category figure...")
    ORIG_CAT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.0), squeeze=False)
    h.plot_category_row(axes[0], embeddings, original_cats, titles=titles)
    fig.tight_layout()
    fig.savefig(ORIG_CAT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {ORIG_CAT}")

    print("Writing energy/atom figure...")
    _save_energy_figure(ORIG_E, embeddings, energy_per_atom, titles)

    print("Writing phase-category figure...")
    PHASE_CAT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.0), squeeze=False)
    h.plot_phase_category_row(axes[0], embeddings, phase_cats, titles=titles)
    fig.tight_layout()
    fig.savefig(PHASE_CAT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PHASE_CAT}")

    # Remove legacy duplicate if present
    legacy = FIG_DIR / "energy_per_atom" / "umap_by_energy_phase_panel.png"
    if legacy.is_file():
        legacy.unlink()
        print(f"Removed {legacy}")

    print("\nDone.")
    for path in (ORIG_CAT, ORIG_E, PHASE_CAT):
        print(f"  {path}")
    print("Phase counts:", {c: phase_cats.count(c) for c in PHASE_CATEGORIES})


if __name__ == "__main__":
    main()
