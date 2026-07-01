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
├── descriptors/
│   ├── behler/          Behler-Parrinello symmetry functions (maml)
│   ├── bispectrum/      Bispectrum coefficients (maml + LAMMPS)
│   ├── chimes/          ChIMES fingerprints (pre-computed descriptor file)
│   ├── euler/           Euler characteristic curves (topological, numpy + ase)
│   └── soap/            SOAP descriptors (dscribe)
├── analysis/
│   └── normalized_category_composition.py   Main composition analysis figure
├── figures/
│   └── normalized_category_composition.png  Current output figure
├── random_baseline.py   Descriptor-free random sampling (baseline comparison)
└── archive/             Old/superseded scripts and figures (not in active use)
```

> **Not committed (see `.gitignore`):** `data.json`, all `replicates_structure_pruning_modified/`
> output directories, `frames_descriptors.pkl`, `natoms.txt`.

---

## Quick Start

### 1. Dependencies

```bash
pip install numpy scikit-learn skmatter monty dscribe maml
```

LAMMPS is required for **Behler-Parrinello** and **Bispectrum** descriptors via `maml`.
Install LAMMPS and update the `os.environ['PATH']` line at the top of those scripts,
or ensure `lmp` is already on your system `PATH`.

### 2. Place the dataset

Copy `data.json` (pymatgen-serialized Si structures) to the project root.

### 3. Run FPS sampling per descriptor

Each script is run from **inside** its own directory:

```bash
# SOAP
cd descriptors/soap
python compute_and_prune.py

# Behler-Parrinello
cd descriptors/behler
python compute_and_prune.py

# Bispectrum
cd descriptors/bispectrum
python compute_and_prune.py

# ChIMES (requires pre-processing step first)
cd descriptors/chimes
python process_raw_descriptors.py   # converts A.txt + natoms.txt → frames_descriptors.pkl
python compute_and_prune.py

# Euler characteristic (topological; no external descriptor library needed)
cd descriptors/euler
python compute_and_prune.py

# Random baseline (run from project root)
python random_baseline.py
```

Each script produces `replicates_structure_pruning_modified/` in its own directory,
containing 10 replicates × 18 pruning ratios (1–90%) of JSON structure files.

### 4. Analyse sampling composition

```bash
python analysis/normalized_category_composition.py
```

Reads the replicate outputs for Behler, Bispectrum, and SOAP and writes
`figures/normalized_category_composition.png`.

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
