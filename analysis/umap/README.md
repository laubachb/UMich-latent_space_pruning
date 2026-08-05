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
│   │   └── data_phase_categories.json   # regenerable
│   └── figures/                           # regenerable PNGs
├── descriptor_cache/
│   ├── README.md
│   └── structure_descriptors.npz          # regenerable, gitignored
└── .venv/
```

Label rules: `analysis/categorization.py`

Step-by-step generation of each file:
[`unsupervised/README.md`](unsupervised/README.md) and
[`descriptor_cache/README.md`](descriptor_cache/README.md).

## Commands (summary)

From repo root (`uv sync` once first):

```bash
# 1) Phase-labeled dataset → unsupervised/data/data_phase_categories.json
uv run python analysis/umap/unsupervised/scripts/relabel_phase_categories.py

# 2) Descriptor cache (.npz) + UMAP figures
#    (needs data.json; for ChIMES also frames_descriptors.pkl; LAMMPS for Behler/Bispectrum)
uv run python analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py --force-recompute

# Later runs can reuse the cache (omit --force-recompute):
uv run python analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py
```
