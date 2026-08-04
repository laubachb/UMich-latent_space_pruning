# Descriptor-Induced Sampling Bias in MLIPs

Systematic comparison of how different atomic descriptors induce sampling biases
when curating training sets via Farthest Point Sampling (FPS). Four descriptor
types are compared on identical silicon datasets, and their effect on downstream
MLIP performance is quantified in low-data regimes. Work out of Develop branch.

**Descriptors:** SOAP · Behler-Parrinello · Bispectrum · ChIMES · Euler Characteristic

---

## Repository Layout

```
.
├── descriptors/         SOAP / Behler / Bispectrum / ChIMES / Euler + FPS pruning
├── pruning/             Random baseline + normalized category composition plots
├── clustering/          Latent-space information & physics-encoding analyses
├── analysis/
│   ├── categorization.py   Shared structure / phase label rules
│   └── umap/               Unsupervised UMAP study (viz only; see analysis/umap/)
├── figures/             Composition + clustering figures
├── models/              Downstream MLIP notes
└── archive/             Old/superseded scripts and figures (not in active use)
```

> **Not committed (see `.gitignore`):** `data.json`, all `replicates_structure_pruning_modified/`
> output directories, `frames_descriptors.pkl`, `natoms.txt`, regenerable descriptor
> caches (`*.npy` / UMAP `structure_descriptors.npz`).

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
cd descriptors/euler       && uv run python compute_and_prune.py && cd ../..
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

### 5. Unsupervised UMAP visualization

Baseline unsupervised UMAPs (category / phase / energy overlays) live under
`analysis/umap/unsupervised/`. UMAP is for visualization only; FPS still runs in
scaled high-dimensional descriptor space. Details and prerequisites:
[`analysis/umap/unsupervised/README.md`](analysis/umap/unsupervised/README.md).

```bash
# Prefer a precomputed structure-level cache (see analysis/umap/descriptor_cache/)
uv run python analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py

# Optional: rebuild phase-labeled dataset overlay
uv run python analysis/umap/unsupervised/scripts/relabel_phase_categories.py
```

---

## Descriptor Details

| Descriptor | Library | Notes |
|---|---|---|
| SOAP | `dscribe` | `r_cut=5.0 Å`, `n_max=9`, `l_max=9` |
| Behler-Parrinello | `maml` + LAMMPS | `cutoff=5.5 Å`, two-body + angular terms |
| Bispectrum | `maml` + LAMMPS | `rcutfac=4.9`, `twojmax=8` |
| ChIMES | pre-computed | Requires `A.txt` from a ChIMES calculation |
| Euler Characteristic | `numpy` + `ase` | Topological ECC, `R_MAX=6.0 Å`, `N_BINS=64` |

SOAP, Behler-Parrinello, Bispectrum and ChIMES are *per-atom* descriptors,
aggregated to structure level via **mean + std** concatenation, then standardized
before FPS. The Euler characteristic is an intrinsically *structure-level*
topological quantity, so no atomic aggregation is applied — the per-frame Euler
characteristic curve is standardized directly. Ten replicates with different random
seeds are run per descriptor to quantify variability.

---

## ChIMES Descriptor Workflow

ChIMES descriptors are generated externally and require two raw files:

- `descriptors/chimes/A.txt` — atomic descriptor matrix from a ChIMES run
- `descriptors/chimes/natoms.txt` — number of atoms per frame

Run `process_raw_descriptors.py` once to convert these into
`frames_descriptors.pkl`, then run `compute_and_prune.py`.

---

## Euler Characteristic Workflow

The Euler characteristic descriptor is computed directly from atomic geometry — no
external descriptor library or pre-processing step is required, only `numpy` and
`ase` (already needed for SOAP).

For each frame, a periodic **Vietoris–Rips filtration** is built over the atomic
point cloud using minimum-image distances, and the Euler characteristic

```
chi(r) = V - E(r) + T(r)
```

is recorded on a fixed radius grid, where `V` = number of atoms, `E(r)` = atom
pairs within `r`, and `T(r)` = triangles whose three edges are all within `r`.
Sampling `chi(r)` on the grid gives one **Euler characteristic curve (ECC)** per
frame — a comparable, structure-level feature vector that is standardized and fed
to FPS exactly like the other descriptors.

Key parameters at the top of `descriptors/euler/compute_and_prune.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `R_MAX` | `6.0` Å | Largest filtration radius (spans several Si coordination shells) |
| `N_BINS` | `64` | Number of radii sampled → descriptor dimension |
| `NORMALIZE_PER_ATOM` | `True` | Divide `chi(r)` by atom count so mixed cell sizes are comparable |

Run from its directory:

```bash
cd descriptors/euler
python compute_and_prune.py
```

Because the ECC is deterministic, the descriptors are computed once and reused
across all 10 FPS replicates.

---

## Data Format

`data.json` is a monty-serialized list of dicts, each with:
- `"structure"` — pymatgen `Structure` object
- `"description"` — string label (used for category analysis)
- `"frame_index"` — integer index (used by ChIMES to match descriptors)
- `"energy"`, `"forces"`, `"stress"` — DFT reference labels
