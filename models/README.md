# Potential Fitting on Pruned Datasets

This directory holds the generation scripts that fit machine-learned interatomic
potentials (MLIPs) on the **pruned silicon training sets** produced upstream by
Farthest Point Sampling (FPS) in [`../descriptors/`](../descriptors/). Four
potential families are trained:

| Potential | Full name | Native representation | Pruned data source |
|---|---|---|---|
| **ChIMES** | Chebyshev Interaction Model for Efficient Simulation | ChIMES many-body polynomials | `../descriptors/chimes/replicates_structure_pruning_modified/` |
| **GAP** | Gaussian Approximation Potential | SOAP kernel | `../descriptors/soap/replicates_structure_pruning_modified/` |
| **NNP** | Neural Network Potential (Behler–Parrinello) | BP symmetry functions | `../descriptors/behler/replicates_structure_pruning_modified/` |
| **SNAP** | Spectral Neighbor Analysis Potential | Bispectrum coefficients | `../descriptors/bispectrum/replicates_structure_pruning_modified/` |

## Motivation

The central question of this repository is **descriptor-induced sampling bias**:
different atomic descriptors, used as the distance metric for FPS, curate
*systematically different* training subsets from the *same* pool of DFT frames
(see the normalized-composition and latent-space analyses in
[`../pruning/`](../pruning/) and [`../clustering/`](../clustering/)). The obvious
follow-up question is: **does that sampling bias actually matter for the fitted
potential?**

To answer it fairly, each potential is fit on the cut curated by **its own native
representation** — the representation the descriptor and the model share:

- SOAP curates the data → **GAP** (a SOAP-kernel model) is trained on it,
- BP symmetry functions curate the data → **NNP** is trained on it,
- Bispectrum curates the data → **SNAP** is trained on it,
- ChIMES curates the data → a **ChIMES** potential is trained on it.

This closes the loop: the descriptor that decides *what* the model sees is the
same descriptor the model uses to *represent* what it sees. Comparing the four
potentials' accuracy as a function of pruning ratio (1–90%), especially in the
**low-data regime**, isolates how each representation's sampling bias propagates
into downstream MLIP quality. A random-selection baseline
([`../pruning/random_baseline.py`](../pruning/random_baseline.py)) anchors the
comparison.

## Layout (planned)

Each potential gets its own subdirectory containing the fit driver plus any
templates/configs it needs. Every script consumes the pruned `*.json` cuts and
writes fitted potentials + validation metrics per (pruning ratio × replicate):

```
models/
├── chimes/     ChIMES fit (chimes_lsq / DLARS solve)
├── gap/        GAP fit (gap_fit / QUIP)
├── nnp/        Behler–Parrinello NN (n2p2 / runner)
├── snap/       SNAP fit (FitSNAP / LAMMPS)
└── README.md
```

Each cut is identified by the same filename convention used upstream:
`si_structures_<descriptor>_mean_std_<PERCENT>percent_replicate<NN>.json`, so a
fit for a given ratio/replicate maps 1:1 to the exact frames FPS selected.

## Reproducibility — seeds are mandatory

**Every script in this repository must fix its random seeds**, and the model
generation scripts here are no exception. Any nondeterminism must be pinned:

- **Data selection** is already seeded upstream: `compute_and_prune.py` sets
  `random.seed(42)` and derives one FPS `random_state` per replicate, so the ten
  replicate cuts are byte-for-byte reproducible.
- **Model fitting** must fix every seed it introduces — weight initialization and
  batch shuffling (NNP), sparse-point / active-set selection (GAP), any
  stochastic solver or regularization-path randomization (SNAP, ChIMES DLARS),
  and train/validation splitting. Record the seed in the output metrics file
  alongside the cut identifier so a run can be replayed exactly.
- **Convention:** default to seed `42` unless a script needs a documented seed
  sweep; when sweeping, log every seed used.

When adding a fit driver, state its seeds explicitly at the top of the script and
echo them into the run log, mirroring the upstream FPS scripts.
