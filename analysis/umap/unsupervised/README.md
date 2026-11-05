# Unsupervised baseline UMAPs

Fully unsupervised UMAP of Si descriptor spaces (Behler / SOAP / Bispectrum / ChIMES).
**All category and energy colors are post-hoc overlays** — UMAP and FPS never see labels or energies.

Baseline settings: `init=spectral`, `min_dist=0.9`, `n_neighbors=15`, `metric=euclidean`, `random_state=42`.

UMAP is visualization only; FPS pruning runs in scaled high-$D$ space.

---

## Files you generate (step by step)

All commands below are run from the **repo root**.

### Step 0 — Environment

```bash
uv sync
```

(or a local `analysis/umap/.venv` with `numpy scipy matplotlib scikit-learn pymatgen ase monty dscribe maml umap-learn`)

### Step 1 — Input structures (`data.json`)

| Output | Path | In git? |
|--------|------|---------|
| Si dataset | `data.json` (repo root) | yes |

Place/keep the 214-structure pymatgen JSON at the repo root. Nothing to run.

### Step 2 — ChIMES pickle (only if you need the ChIMES UMAP panel)

| Output | Path | In git? |
|--------|------|---------|
| Frame matrices | `descriptors/chimes/frames_descriptors.pkl` | no (~70 MB) |

```bash
# Prerequisites (gitignored): descriptors/chimes/A.txt, descriptors/chimes/natoms.txt
cd descriptors/chimes
uv run python process_raw_descriptors.py
cd ../..
```

Skip this step only if you already have a full `structure_descriptors.npz` that
includes the `chimes` key (see Step 4).

### Step 3 — Phase labels JSON

| Output | Path | In git? |
|--------|------|---------|
| Labeled dataset | `analysis/umap/unsupervised/data/data_phase_categories.json` | yes (also regenerable) |

```bash
uv run python analysis/umap/unsupervised/scripts/relabel_phase_categories.py
```

Reads `data.json`, applies `analysis/categorization.py` (`category` +
`phase_category`), writes the JSON above.

### Step 4 — Structure-level descriptor cache (`.npz`)

| Output | Path | In git? |
|--------|------|---------|
| Descriptor cache | `analysis/umap/descriptor_cache/structure_descriptors.npz` | **no** (gitignored) |

```bash
# Also needs LAMMPS on PATH for Behler + Bispectrum
uv run python \
  analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py --force-recompute
```

Built by `scripts/umap_helpers.py` → `load_or_compute_descriptors`:

1. Load structures from `data.json`
2. Compute SOAP / Behler / Bispectrum / ChIMES features
3. Aggregate each structure with **mean ‖ std**
4. Save keys `behler`, `soap`, `bispectrum`, `chimes` via `numpy.savez`

Details (shapes, SOAP params, ChIMES pooling):
[`../descriptor_cache/README.md`](../descriptor_cache/README.md).

Without `--force-recompute`, an existing `.npz` is loaded and missing keys only
are filled in.

### Step 5 — UMAP figures (PNGs)

| Output | Path | In git? |
|--------|------|---------|
| Legacy categories | `figures/original_categories/umap_by_category.png` | yes |
| Phase categories | `figures/phase_categories/umap_by_phase.png` | yes |
| Energy / atom | `figures/energy_per_atom/umap_by_energy.png` | yes |

Same command as Step 4 (the generator writes figures after descriptors):

```bash
uv run python analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py
```

Uses `data_phase_categories.json` when present (Step 3). Reuses the `.npz`
from Step 4 unless you pass `--force-recompute`.

---

## Prerequisites summary

| Path / tool | Role | In git? |
|-------------|------|---------|
| `data.json` | Structures + DFT energies | yes |
| `analysis/categorization.py` | Label rules | yes |
| `analysis/umap/unsupervised/scripts/*.py` | Generators + helpers | yes |
| `analysis/umap/descriptor_cache/structure_descriptors.npz` | Structure-level descriptors | **no** — generate (Step 4) |
| `descriptors/chimes/frames_descriptors.pkl` | Raw ChIMES frames | **no** — generate (Step 2) |
| LAMMPS on `PATH` | Behler / Bispectrum recompute | system |

### Python packages

Prefer repo-root **uv** (`pyproject.toml` / `uv.lock`, includes `umap-learn`):

```bash
uv sync
```

### Other system prerequisites

| Requirement | Needed when |
|-------------|-------------|
| **Python ≥ 3.12** (uv pin) | Always with `uv sync` |
| **LAMMPS** (`lmp_serial` / `lmp`) | Step 4 without a complete `.npz` that already has Behler/Bispectrum |
| **matplotlib Agg** | Writing PNGs (headless OK) |

### What this directory ships in git

| Path | Content |
|------|---------|
| `scripts/relabel_phase_categories.py` | Step 3 |
| `scripts/generate_unsupervised_umaps.py` | Steps 4–5 |
| `scripts/umap_helpers.py` | Descriptor / UMAP / plot helpers |
| `data/data_phase_categories.json` | Step 3 output (can regenerate) |
| `figures/...` | Step 5 output (can regenerate) |
| `README.md` | This file |

---

## Quick start (all steps)

```bash
# 0) Environment
uv sync

# 1) data.json already at repo root

# 2) ChIMES pickle (if building ChIMES from scratch)
cd descriptors/chimes && uv run python process_raw_descriptors.py && cd ../..

# 3) Phase labels
uv run python analysis/umap/unsupervised/scripts/relabel_phase_categories.py

# 4–5) Descriptor .npz + UMAP figures
uv run python \
  analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py --force-recompute
```

---

## Data preprocessing

**Per structure**

1. Atomic features $N \times d$ (or $3N \times d$ for ChIMES force-basis rows)
2. mean $\Vert$ std aggregation: $\mathbf{x}=[\bar{\mathbf{v}},\,\mathrm{std}(\mathbf{v})]$ → length $2d$
3. One structure vector $\mathbf{x}\in\mathbb{R}^{2d}$

**Full dataset (214 structures)**

1. Stack → $214 \times 2d$ feature matrix
2. `StandardScaler` (column-wise: mean 0, variance 1)
3. UMAP → $214 \times 2$ (plot only)

```text
Atomic features (N x d, or 3N x d for ChIMES)
        |
        v
mean || std  -->  one structure vector x in R^{2d}
        |
        v
stack all frames  -->  214 x 2d
        |
        v
StandardScaler (mean 0, var 1)
        |
        v
UMAP  -->  214 x 2  (plot only)
```

Feature widths $d$: Behler 27, Bispectrum 56, ChIMES 231, SOAP 324  
(structure-level $D=2d$ after mean||std).

---

## Original categories

![UMAP by original category](figures/original_categories/umap_by_category.png)

- **Normal (300K) vs High Temp / Other**
  - SOAP / Bispectrum isolate 300 K crystal; mid-$T$ (Other) and liquid-like (High Temp) frames leave the diamond multipole pattern
  - Behler still separates Normal but with less margin
  - ChIMES keeps Normal nearer the mixed hot/defect cloud
- **Strained** — elastic modes change both $r$ and $\theta$
  - Angular descriptors (Behler / SOAP / Bispectrum) split strained cells into several islands (mode / magnitude)
  - ChIMES merges them toward a continuum of distorted crystals
- **Vacancy (300K)** — local undercoordination is an angular / coordination signal
  - Well isolated in SOAP / Bispectrum / Behler
  - In ChIMES it bleeds into High Temp / Other via shared broad pair correlation
- **Surface** — incomplete neighbor shell
  - Strong odd multipoles in SOAP / Bispectrum
  - For ChIMES, missing pairs change force-basis rows, which can resemble other defective / low-coordination frames after mean||std pooling

> This Si dataset’s manmade categories are largely **angular-order** categories: perfect tetrahedral crystal, strained crystal, melt, surface, vacancy.

---

## Phase categories

![UMAP by phase category](figures/phase_categories/umap_by_phase.png)

### Strained vs unstrained diamond

- **Unstrained Diamond** = ground state + normal AIMD at $T \le 1518\,\mathrm{K}$ (still solid / tetrahedral by coordination-number diagnostics)
- **Strained Diamond** = elastic $2\times2\times2$ mode/strain cells
- All three local fingerprints (Behler, SOAP, Bispectrum) pull Strained Diamond off Unstrained Diamond
  - Strain lowers symmetry ($Fd\bar{3}m$ → a lower-symmetry space group) and splits nearest-neighbor distances/angles
- **Within Strained**
  - Bispectrum resolves the most mode / magnitude islands
  - SOAP and Behler look more alike: fewer separations, coarser grouping of strain cells
- **Why ChIMES softens the split**
  - The force design matrix varies smoothly with $r_{ij}$; strain is a continuous path from the unstrained crystal in that metric (and we never had invariant per-atom fingerprints to begin with)

### Liquid, vacancies, surface

- **Liquid** (2530 / 3374 K normal AIMD): loss of orientational order + CN rise (~5.5). SOAP / Bispectrum respond strongly to isotropy of $\rho(\mathbf{r})$; ChIMES force rows mainly track broadened pair statistics — closer to other disordered frames after pooling
- **Vacancy-3374K** near diamond, not Liquid: snapshots remain CN ≈ 4 (crystal-like) and sit at low energy/atom. Angular per-atom descriptors correctly treat them as defective solids
- **Vacancy-300K**: clearer defect signal (undercoordinated neighbors) and high energy/atom; own island for Behler / SOAP / Bispectrum; ChIMES often pulls it toward Liquid via shared distance / force-basis patterns
- **Surface**: missing half-space of neighbors → high anisotropy for density expansions (SOAP / Bispectrum); for ChIMES it is a change in which force-basis rows are active, more easily confused with other defects after mean||std

---

## DFT potential energy / atom

![UMAP colored by energy per atom](figures/energy_per_atom/umap_by_energy.png)

Same unsupervised embedding as the category maps; only the color changes.

- **High-$E$** aligns with **Liquid** and **Vacancy-300K**
- **Low-$E$** aligns with **Unstrained Diamond**, **Strained Diamond**, and **Vacancy-3374K**
- Here **temperature ≠ energy** (not thermo energy): Vacancy-3374K is low-$E$; Vacancy-300K is high-$E$ (denser cells / higher PotEng)
- **Behler / SOAP / Bispectrum**: several purple (low-$E$) islands = distinct low-$E$ structures (crystal vs strain modes vs hot vacancy) — **structure**, not energy, drives the split
- **ChIMES**: 2-part cloud; force-basis / PotEng-aligned split — consistent with a representation built for forces, not angular-order categories

---

## Thoughts (FPS implications)

These are **hypotheses** suggested by UMAP (2-D manifold, not raw high-$D$ distances) + descriptor functional forms. Confirm with FPS category-composition curves.

- FPS / UMAP use **descriptor distances only** — no energies, no category labels
- FPS in SOAP / Bispectrum space may **oversample** rare angular environments (surfaces, distinct strain modes) because they sit far from the diamond core in those metrics — matching the sharp UMAP islands
- FPS in ChIMES space may **undersample** Surface (buried in the large mixed cloud) and **oversample** Vacancy-300K (in the smaller high-$E$ cloud with Liquid), matching the diffuse / two-cloud UMAP
- Interpret cautiously: ChIMES force-basis space is not the same geometric object as SOAP space
