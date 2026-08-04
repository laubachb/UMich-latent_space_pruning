# UMAP analysis

Visualizations of Si descriptor spaces (Behler / SOAP / Bispectrum / ChIMES).
UMAP is for visualization only; FPS pruning runs in scaled high-D space.

## Layout

```
analysis/umap/
├── unsupervised/                 # baseline unsupervised study
│   ├── scripts/
│   │   ├── generate_unsupervised_umaps.py
│   │   ├── relabel_phase_categories.py
│   │   └── umap_helpers.py
│   ├── data/
│   │   └── data_phase_categories.json
│   └── figures/
│       ├── original_categories/  # legacy labels
│       ├── phase_categories/     # phase labels
│       └── energy_per_atom/      # DFT energy/atom overlays
├── descriptor_cache/             # structure_descriptors.npz (+ README)
└── .venv/
```

Label rules: `analysis/categorization.py`

## Commands

From repo root (preferred: `uv sync` once, then `uv run`):

```bash
# Rebuild phase-labeled dataset
uv run python analysis/umap/unsupervised/scripts/relabel_phase_categories.py

# Baseline unsupervised UMAPs (category + energy overlays)
uv run python analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py
```

A local `analysis/umap/.venv` also works if you prefer not to use the repo-root uv env.

Requires `data.json` at the repo root. LAMMPS (`lmp_serial`) on `PATH` is only
needed when recomputing Behler/Bispectrum without a descriptor cache.
