# UMAP analysis

Visualizations of the Si latent spaces of the manuscript (SOAP, Behler–Parrinello,
bispectrum, MTP, ChIMES, ChIMES fingerprint, Euler characteristic) coloured by
structure category, phase and energy. UMAP is for visualization only; FPS runs in the
standardised high-dimensional spaces. The descriptors are the `lsp` caches
(`results/Si/descriptors/`), i.e. the same matrices as the manuscript's analyses.

## Layout

```
analysis/umap/
└── unsupervised/                 # baseline unsupervised study
    ├── scripts/
    │   ├── generate_unsupervised_umaps.py
    │   ├── relabel_phase_categories.py
    │   └── umap_helpers.py       # loads the lsp latent spaces; UMAP + plotting helpers
    ├── data/
    │   └── data_phase_categories.json   # labels only (regenerable)
    └── figures/                           # regenerable PNGs
```

Label rules: `analysis/categorization.py`

Step-by-step: [`unsupervised/README.md`](unsupervised/README.md).

## Commands (summary)

From repo root (`uv sync` and `python scripts/fetch_mlearn_data.py Si` once first):

```bash
# 1) Phase labels → unsupervised/data/data_phase_categories.json (optional record)
python analysis/umap/unsupervised/scripts/relabel_phase_categories.py

# 2) UMAP figures (descriptors from the lsp cache; computed on first use)
python analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py
```
