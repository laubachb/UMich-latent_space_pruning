# Descriptor cache

Local store for **structure-level** descriptors used by UMAP (after mean‖std
aggregation). `structure_descriptors.npz` is committed so clones can regenerate
figures without re-running SOAP / LAMMPS / ChIMES.

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

## How it is used

1. `unsupervised/scripts/umap_helpers.py` → `load_or_compute_descriptors`
2. If this file exists, UMAP scripts load it and skip SOAP / LAMMPS / ChIMES pickle I/O
3. Rebuild with `--force-recompute` on the generator (needs packages + LAMMPS / ChIMES inputs as appropriate)

## Create / refresh

From repo root:

```bash
uv run python \
  analysis/umap/unsupervised/scripts/generate_unsupervised_umaps.py --force-recompute
```
