#!/usr/bin/env python3
"""
Regenerate the unsupervised baseline UMAP figures (category, phase and energy overlays).

One panel per latent space of the manuscript, in this order: SOAP, Behler-Parrinello,
bispectrum and MTP (per-atom mean), ChIMES (A-matrix energy row / N), ChIMES
fingerprint (full frame, alpha = 1) and the Euler characteristic curve. The
descriptors come from the lsp cache (results/Si/descriptors/), so these panels embed
exactly the matrices behind the manuscript's analyses. Colours are post-hoc overlays:
UMAP never sees labels or energies.

Writes under analysis/umap/unsupervised/figures/:
  original_categories/umap_by_category.png
  phase_categories/umap_by_phase.png
  energy_per_atom/umap_by_energy.png

Run from the repo root:
    python analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py
    python analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py --force-recompute
(--force-recompute rebuilds the lsp descriptor caches first.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
UNSUP_DIR = SCRIPT_DIR.parent
FIG_DIR = UNSUP_DIR / "figures"

sys.path.insert(0, str(SCRIPT_DIR))
import umap_helpers as h  # noqa: E402
from categorization import (  # noqa: E402
    PHASE_CATEGORIES,
    categorize_phase,
    categorize_structure,
)

ORIG_CAT = FIG_DIR / "original_categories" / "umap_by_category.png"
ORIG_E = FIG_DIR / "energy_per_atom" / "umap_by_energy.png"
PHASE_CAT = FIG_DIR / "phase_categories" / "umap_by_phase.png"
NCOL = 4


def _grid(n_panels: int):
    """(fig, panel axes, spare axis) for n_panels on a NCOL-column grid plus one spare cell."""
    nrow = -(-(n_panels + 1) // NCOL)
    fig, axes = plt.subplots(nrow, NCOL, figsize=(4.2 * NCOL, 3.9 * nrow), squeeze=False)
    flat = list(axes.ravel())
    for ax in flat[n_panels:]:
        ax.axis("off")
    return fig, flat[:n_panels], flat[n_panels]


def main() -> None:
    frames = h.load_frames()
    descriptions = [d.get("description") or "" for d in frames]
    original_cats = [categorize_structure(desc) for desc in descriptions]
    phase_cats = [categorize_phase(desc, d.get("group")) for desc, d in zip(descriptions, frames)]
    energy_per_atom = np.array([d["outputs"]["energy"] / len(d["structure"]) for d in frames])

    print("Loading latent spaces from the lsp cache...")
    descriptors = h.load_descriptors(force_recompute="--force-recompute" in sys.argv)
    names = h._panel_order(descriptors)
    titles = {k: k for k in names}

    print("Computing unsupervised baseline UMAP embeddings...")
    embeddings = {name: h.compute_umap_embedding(X, init="spectral") for name, X in descriptors.items()}

    print("Writing original-category figure...")
    fig, axes, spare = _grid(len(names))
    h.plot_category_row(axes, embeddings, original_cats, titles=titles, legend_ax=spare)
    fig.tight_layout()
    ORIG_CAT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ORIG_CAT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {ORIG_CAT}")

    print("Writing phase-category figure...")
    fig, axes, spare = _grid(len(names))
    h.plot_phase_category_row(axes, embeddings, phase_cats, titles=titles, legend_ax=spare)
    fig.tight_layout()
    PHASE_CAT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PHASE_CAT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PHASE_CAT}")

    print("Writing energy/atom figure...")
    fig, axes, spare = _grid(len(names))
    h.plot_descriptor_row(axes, embeddings, energy_per_atom, "Energy / atom (eV)", titles=titles,
                          shared_colorbar=False)
    fig.tight_layout()
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=float(energy_per_atom.min()),
                                                                   vmax=float(energy_per_atom.max())))
    sm.set_array([])
    box = spare.get_position()
    cax = fig.add_axes([box.x0 + 0.35 * box.width, box.y0 + 0.1 * box.height, 0.08 * box.width, 0.8 * box.height])
    fig.colorbar(sm, cax=cax, label="Energy / atom (eV)")
    ORIG_E.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ORIG_E, dpi=200)
    plt.close(fig)
    print(f"Wrote {ORIG_E}")

    print("\nDone.")
    for path in (ORIG_CAT, ORIG_E, PHASE_CAT):
        print(f"  {path}")
    print("Phase counts:", {c: phase_cats.count(c) for c in PHASE_CATEGORIES})


if __name__ == "__main__":
    main()
