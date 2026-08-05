# Descriptor cache

Local store for **structure-level** descriptors used by UMAP (after mean‖std
aggregation). The `.npz` is **gitignored** — generate it locally (see below).

## Expected file

`structure_descriptors.npz` — NumPy archive with one array per method:

| Key | Shape (this Si set) | Meaning |
|-----|---------------------|---------|
| `behler` | `(214, 54)` | Behler–Parrinello (27 × 2) |
| `soap` | `(214, 648)` | SOAP (324 × 2) |
| `bispectrum` | `(214, 112)` | Bispectrum / SNAP-style (56 × 2) |
| `chimes` | `(214, 462)` | ChIMES force-basis rows pooled (231 × 2) |

Each row is one structure; columns are concatenated mean ‖ std of per-atom
(or per force-row) features.

## Step-by-step: generate `structure_descriptors.npz`

All commands from the **repo root**.

### 1. Environment

```bash
uv sync
```

### 2. Structures

Ensure `data.json` is at the repo root (214 Si structures).

### 3. ChIMES input (needed for the `chimes` array)

Place ChIMES raw files, then build the pickle once:

```bash
# required (gitignored): descriptors/chimes/A.txt, descriptors/chimes/natoms.txt
cd descriptors/chimes
uv run python process_raw_descriptors.py   # writes frames_descriptors.pkl
cd ../..
```

### 4. Compute and save the `.npz`

```bash
uv run python \
  analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py --force-recompute
```

What this does (via `umap_helpers.load_or_compute_descriptors`):

1. Load geometries from `data.json`
2. Compute features:
   - **SOAP** — `dscribe` (`r_cut=5.1`, `n_max=8`, `l_max=8`)
   - **Behler** — `maml` + LAMMPS on `PATH`
   - **Bispectrum** — `maml` + LAMMPS on `PATH`
   - **ChIMES** — aggregate rows from `frames_descriptors.pkl`
3. Aggregate each structure with **mean ‖ std** → `(214, 2d)`
4. Write `analysis/umap/descriptor_cache/structure_descriptors.npz`
5. Also regenerate the unsupervised UMAP PNG figures

Without `--force-recompute`, the script **loads** an existing `.npz` if present
and only computes missing keys.

### Requirements for a full four-key cache

| Method | Needs |
|--------|--------|
| SOAP | `dscribe` (from `uv sync`) |
| Behler / Bispectrum | LAMMPS (`lmp_serial` / `lmp`) on `PATH` |
| ChIMES | `descriptors/chimes/frames_descriptors.pkl` |

## How it is used

1. `unsupervised/scripts/umap_helpers.py` → `load_or_compute_descriptors`
2. If the `.npz` exists, UMAP skips expensive recomputation
3. Arrays are `StandardScaler`-normalized, then embedded with UMAP
