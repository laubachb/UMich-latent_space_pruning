# Descriptor-Induced Sampling Bias in MLIPs

Systematic comparison of how different atomic descriptors induce sampling biases
when curating training sets via Farthest Point Sampling (FPS). Four descriptor
types are compared on identical silicon datasets, and their effect on downstream
MLIP performance is quantified in low-data regimes. Work out of Develop branch.

**Descriptors:** SOAP · Behler-Parrinello · Bispectrum · ChIMES

---

## Repository Layout

```
.
├── descriptors/
│   ├── behler/          Behler-Parrinello symmetry functions (maml)
│   ├── bispectrum/      Bispectrum coefficients (maml + LAMMPS)
│   ├── chimes/          ChIMES fingerprints (pre-computed descriptor file)
│   └── soap/            SOAP descriptors (dscribe)
├── analysis/
│   └── normalized_category_composition.py   Main composition analysis figure
├── figures/
│   └── normalized_category_composition.png  Current output figure
├── random_baseline.py   Descriptor-free random sampling (baseline comparison)
└── archive/             Old/superseded scripts and figures (not in active use)
```

> **Not committed (see `.gitignore`):** `data.json`, all `replicates_structure_pruning_modified/`
> output directories, ChIMES A matrices (`A.txt`), `natoms.txt`, `chimes_raw.npz`.

---

## Quick Start

### 1. Environment (uv)

The entire Python stack is pinned in [`pyproject.toml`](pyproject.toml) and frozen in
[`uv.lock`](uv.lock), so [uv](https://docs.astral.sh/uv/) reproduces the exact
environment that produced the committed figures:

```bash
uv sync          # creates .venv/ from uv.lock (bit-for-bit reproducible)
```

Then run any script with `uv run python <script>` (no manual `activate` needed).
The pinned Python is `3.13` (see `.python-version`); any `3.12`–`3.13` works.

> **HPC note — constrained `$HOME`:** uv caches wheels under `~/.cache/uv` by
> default. If your home directory is quota-limited, point the cache at scratch:
> `export UV_CACHE_DIR=/path/with/space/.uv-cache` before `uv sync`.

**External prerequisites (not handled by uv):**
- **`data.json`** — pymatgen-serialized Si structures — placed at the project root.
- **LAMMPS** — `lmp` (or `lmp_serial`) on `PATH`, required only for the
  **Behler-Parrinello** and **Bispectrum** descriptors (maml calls it as a subprocess).
- **ChIMES raw files** — `descriptors/chimes/A.txt` + `natoms.txt`, required only for
  the **ChIMES** descriptor.

### 2. Run FPS sampling per descriptor

Each script runs from **inside** its own directory and writes
`replicates_structure_pruning_modified/` there (10 replicates × 18 pruning ratios,
1–90%). Each script caches its raw descriptor matrix as
`structure_descriptors_*.npy` in its folder, so re-runs skip the expensive
descriptor evaluation (delete the `.npy` to force a rebuild).

```bash
cd descriptors/soap        && uv run python compute_and_prune.py && cd ../..
cd descriptors/behler      && uv run python compute_and_prune.py && cd ../..   # needs LAMMPS
cd descriptors/bispectrum  && uv run python compute_and_prune.py && cd ../..   # needs LAMMPS
cd descriptors/chimes && uv run python process_raw_descriptors.py \
                       && uv run python compute_and_prune.py && cd ../..       # A.txt + natoms.txt

# Descriptor-free random baseline
uv run python pruning/random_baseline.py
```

### 3. Analyse sampling composition

```bash
uv run python pruning/normalized_category_composition.py
```

Reads the Behler / Bispectrum / SOAP / ChIMES replicate outputs and writes
`figures/normalized_category_composition.png`.

```bash
uv run python pruning/quests_information_entropy.py
```

Scores every pruned cut (each FPS latent space + random) in one shared,
descriptor-agnostic atomic space, the QUESTS representation (k=32, 5 Å cutoff,
h=0.015 Å⁻¹). It plots H(cut)/H(full dataset) and the QUESTS coverage of the full
dataset (fraction of atomic environments with δH ≤ 0) against retention %, as
mean ± s.d. over replicates. Writes `figures/quests_information_entropy.png` and
`clustering/cache/quests_entropy_by_retention.csv`.

### 4. Latent-space information & physics analysis

```bash
# Cache one structure-level descriptor matrix per space + a shared metadata table
uv run python clustering/compute_descriptor_matrices.py

# Then any of:
uv run python clustering/normalized_distance_histograms.py   # max-normalized distance distributions
uv run python clustering/participation_ratio.py              # effective dimensionality per space
uv run python clustering/physics_encoding.py                 # decodability of physics per space
uv run python clustering/distance_analysis.py                # distance vs physics (Spearman)
uv run python clustering/latent_embeddings.py                # PCA / t-SNE embeddings
uv run python clustering/cluster_motifs.py                   # KMeans/Ward vs metadata groups
```

Each writes its figure(s) to `figures/` and small result tables to
`clustering/cache/` (the large `*.npy` descriptor matrices there are regenerable
and gitignored).

### 5. Publication figures (gnuplot)

```bash
uv run python plotting/export_gnuplot_data.py     # writes plotting/data/*.dat
cd plotting && gnuplot fig_umap.gp fig_composition.gp fig_information.gp fig_distance.gp
```

300 dpi PNGs land in `figures/` (drawn with the cairo PDF terminal, rasterized by `pdftoppm`):

| Figure | Content |
|---|---|
| `fig_umap_latent_spaces.png` | UMAP (min_dist=0.9, seed 42) of the four standardized structure-level latent spaces, colored by structure category |
| `fig_dataset_composition.png` | Category composition of the pruned sets vs retention (pooled over 10 replicates), random + each FPS latent space |
| `fig_dataset_information.png` | QUESTS (a) diversity D/D_full, (b) coverage, (c) entropy H/H_full vs retention (mean ± s.d.) |
| `fig_distance_analysis.png` | Per latent space: max-normalized interpoint distances stacked by category (top) and each category's mean distance to the other frames vs its FPS enrichment at ≤10 % retention, with Spearman ρ (bottom) |
| `fig_toc.png` | TOC graphic: 2D PCA of each latent space (distance-preserving, r ≥ 0.97) with the FPS 5 % picks circled → GAP / NNP / SNAP / ChIMES → "Nominal Error" panel (**schematic** curves until model results are in) |

Run steps 2–4 first (the exporter reads the FPS cuts, `clustering/cache/*_descriptors.npy`
and `clustering/cache/quests_entropy_by_retention.csv`). Shared fonts and palettes live in
`plotting/style.gp`.

---

## Descriptor Details

| Descriptor | Library | Notes |
|---|---|---|
| SOAP | `dscribe` | `r_cut=5.0 Å`, `n_max=9`, `l_max=9` |
| Behler-Parrinello | `maml` + LAMMPS | `cutoff=5.5 Å`, two-body + angular terms |
| Bispectrum | `maml` + LAMMPS | `rcutfac=4.9`, `twojmax=8` |
| ChIMES | pre-computed | Energy row / natoms from a ChIMES `A.txt` |

SOAP, Behler-Parrinello and Bispectrum are *per-atom* descriptors,
aggregated to structure level via **mean + std** concatenation, then standardized
before FPS. ChIMES uses the per-atom average of its cluster basis sums (the
A-matrix energy row divided by the atom count; see below). Ten replicates with
different random seeds are run per descriptor to quantify variability.

---

## ChIMES Descriptor Workflow

ChIMES descriptors are generated externally and require two raw files:

- `descriptors/chimes/A.txt` — least-squares design matrix (A matrix) from a ChIMES run
- `descriptors/chimes/natoms.txt` — atom count of each A-matrix row's frame (one line per row)

Both are large and gitignored; A matrices must never be committed.

Each frame's block in `A.txt` is 3·natoms force rows followed by 3 identical
energy rows; frames are in `data.json` order (checked by atom count).
`process_raw_descriptors.py` validates this layout and writes `chimes_raw.npz`;
`compute_and_prune.py` then uses **energy row / natoms** as the structure
descriptor. The force rows are deliberately *not* mean+std aggregated: per
parameter they sum to zero over atoms (Newton's third law), so their mean is
identically zero and, after standardization, would contribute only noise.

---

## Data Format

`data.json` is a monty-serialized list of dicts, each with:
- `"structure"` — pymatgen `Structure` object
- `"description"` — string label (used for category analysis)
- `"energy"`, `"forces"`, `"stress"` — DFT reference labels
