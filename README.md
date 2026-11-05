# Descriptor-Induced Sampling Bias in MLIPs

How the latent space of an atomistic descriptor shapes the training sets that
subsampling selects from a fixed pool of DFT frames, and how that propagates into
the fitted potentials. The same analysis runs on the six single-element benchmark
datasets of the cost-and-performance assessment of Zuo et al.
(*J. Phys. Chem. A* **124**, 731 (2020); Ni, Cu, Li, Mo, Si, Ge), each with the
hyperparameters of that paper's expert-optimised models.

**Latent spaces** (one per MLIP family)

| id | descriptor | model | structure-level vector |
|---|---|---|---|
| `soap` | SOAP power spectrum (dscribe) | GAP | mean + std over atoms |
| `behler` | Behler–Parrinello G2 + G4 symmetry functions | NNP | mean + std over atoms |
| `bispectrum` | bispectrum coefficients (maml + LAMMPS `sna/atom`) | SNAP | mean + std over atoms |
| `mtp` | moment-tensor basis functions (level 16–24, MLIP-2 contraction tables) | MTP | mean + std over atoms |
| `chimes` | ChIMES A-matrix energy row / N (per-atom cluster-polynomial sums) | ChIMES | per-atom mean (= design row / N) |
| `chimes_fingerprint` | full-frame ChIMES cluster-graph fingerprint, α = 1 (Laubach et al.) | ChIMES | 3 × 100-bin histograms, no atomic averaging |
| `<id>_mean` | per-atom **mean** only of any atom-centred descriptor above | | design row / N, the aggregation a linear / average-kernel model sees |

**Samplers** (applied identically in every latent space, 10 replicate seeds × 18
retentions 1–9 %, 10–90 %): `fps` farthest point sampling (extreme-seeking, nested
cuts), `kmedoids` k-means medoids (density-representative), `leverage` randomized
CUR by statistical leverage (subspace-spanning), and the descriptor-free `random`
baseline.

**Analyses**: category composition of the cuts vs retention; QUESTS information
(diversity, coverage, entropy) of every cut in one descriptor-agnostic atomic
space; category-resolved distance distributions and their Spearman correlation
with selection enrichment; UMAP/PCA embeddings; effective dimensionality.

---

## Repository layout

```
lsp/                      the package
  config.py data.py paths.py selection.py
  descriptors/            one module per latent space + registry/caching (__init__.py)
  analysis/               composition, quests, distance, embeddings, dimensionality, export
  mtp_levels/             MLIP-2 untrained MTP definitions (contraction tables), BSD-2
configs/<ds>.json         per-dataset hyperparameters (Si, Cu, Ge, Li, Mo, Ni, si_legacy)
configs/local_paths.json  machine-specific executables (gitignored; see .example)
scripts/                  run_pipeline.py, run_dataset.sbatch, fetch_mlearn_data.py, export_cuts.py
plotting/                 gnuplot figure scripts (parametric in the dataset) + make_figures.sh
tests/                    test_legacy_si.py (exact reproduction of the original Si study)
models/                   notes for the downstream potential fits
external/                 UNTRACKED inputs: mlearn/<El>/training.json, chimes/<El>/{A.txt,...}
results/<ds>/             descriptors/ (npy caches, ignored) selection/ (tracked) analysis/ (small csv tracked)
plotting/data/<tag>/      exported .dat files per (dataset, aggregation, sampler) — tracked
figures/<tag>/fig_*.png   publication figures — tracked
```

## Quick start

```bash
uv sync                                        # exact environment from uv.lock (Python 3.13)
python scripts/fetch_mlearn_data.py            # external/mlearn/<El>/training.json
cp configs/local_paths.example.json configs/local_paths.json   # edit: LAMMPS + fingerprint tool
# ChIMES inputs: external/chimes/<El>/{A.txt, b-labeled.txt, dim.txt, train.xyzf, params.txt}
#   (chimes_lsq outputs of T. Sundberg's per-element fits; never committed)

python scripts/run_pipeline.py Cu              # descriptors -> selection -> analysis -> export
cd plotting && ./make_figures.sh Cu Cu_mean Cu_kmedoids Cu_leverage
```

On Dane: `sbatch --export=DS=Cu scripts/run_dataset.sbatch` (pdebug, 1 node; the
QUESTS stage is the slow part, ~20–40 min per dataset; every stage is cached and
resumable, so re-submitting continues where a job stopped). `STAGES="analysis export"`
and `EXTRA="--sampler fps"` can be exported to restrict a job.

Per-cut JSON files for model fitting (one file per retention × replicate, same
format as `training.json`) are not stored; generate them from the selections:

```bash
python scripts/export_cuts.py Cu fps-chimes fps-soap random --out /scratch/cuts/Cu
```

## Descriptor hyperparameters

`configs/<El>.json` carries, for each element, the settings of the published
models of Zuo et al. (read from `mlearn/mvl_models`: `gap.xml`, n2p2 `input.nn`,
`SNAPotential.snapparam`; MTP from SI Table S10):

| | Ni | Cu | Li | Mo | Si | Ge |
|---|---|---|---|---|---|---|
| SOAP cutoff (n_max = l_max = 8, σ = 0.5 Å) | 3.9 | 4.1 | 4.8 | 5.2 | 5.4 | 5.4 |
| BP cutoff (η grid, r_s, ζ per input.nn) | 3.9 | 4.1 | 5.2 | 5.2 | 5.5 | 5.6 |
| SNAP rcutfac (twojmax) | 4.1 (8) | 4.1 (8) | 5.1 (8) | 4.6 (6) | 4.9 (8) | 5.5 (8) |
| MTP cutoff (level) | 4.0 (20) | 3.9 (20) | 5.1 (16) | 5.2 (20) | 4.7 (24) | 5.1 (24) |
| ChIMES orders 2b/3b/4b (Sundberg fits) | 8/14/3 | 18/10/3 | 6/6/0 | 22/10/4 | 6/10/4 | 4/10/4 |

Notes
* **Behler η units.** n2p2/RuNNer take Bohr; mlearn converted r_c and r_s to Bohr but
  wrote η unchanged, so the published NNPs (and SI Tables S1–S6) use
  η = 0.036, 0.071, 0.179 … Å⁻² (= 0.01, 0.02, 0.05 Bohr⁻²). The configs store the
  Å⁻² values. The original Si study used 0.01/0.02/0.05 Å⁻² (3.6× broader
  Gaussians) and SOAP 5.0 Å / n_max = l_max = 9 / σ = 1 Å; `configs/si_legacy.json`
  keeps those so `tests/test_legacy_si.py` reproduces it exactly.
* **MTP radial functions.** MLIP trains the radial functions; a descriptor must be
  model-free, so the basis functions are evaluated with the untrained Chebyshev
  radial basis (identity radial-coefficient matrix), i.e. the linear MTP of the
  original formulation. Contraction tables are MLIP-2's `untrained_mtps/<level>.mtp`.
* **ChIMES A matrix.** `chimes_lsq` with energies fitted: each frame is 3N force rows
  + 3 identical energy rows (`b-labeled.txt` labels, `dim.txt` sizes). Frames follow
  `train.xyzf` = `training.json` order; cells too small for the ChIMES cutoffs were
  replicated for the fit, which leaves energy_row / N unchanged (checked frame by frame).
  Force rows are never aggregated: their atom-mean is identically zero (Newton's 3rd law).
* **Bispectrum** uses per-atom rows (`pot_fit=False`); the SNAP *fitting* rows
  (`pot_fit=True`) carry 1/(3N+1) atom-count factors that made small surface cells
  outliers in an earlier version of the Si study.
* **ChIMES fingerprint** uses the dataset's own ChIMES `params.txt` (same cutoffs,
  S_MINIM and Morse λ as the A matrix), LAMMPS `pair_style chimesFF fingerprint 1`
  and the centrality-corrected `fingerprint_all_alphas` tool at α = 1.

## Structure categories

Parsed from the mlearn descriptions with one rule set: `Surface` (slabs),
`Strained` (strain modes + ground-state/relaxed crystal), `Vacancy <T> K`,
`Bulk <T> K` (AIMD; Li's shrunken/expanded volumes fold into their temperature,
the volume state is kept as metadata). Stacking/colour order: Surface, vacancies
cold→hot, bulk hot→cold, Strained.

## Outputs per dataset

* `results/<ds>/selection/<sampler>-<space>.json`, `random.json` — the selections
  (FPS: ten rankings; others: explicit cuts). Tracked.
* `results/<ds>/analysis/composition.csv`, `dimensionality.csv`, `distance_selection.csv`
  (tracked); `quests.csv`, `distance_histograms.csv`, `embeddings.csv` (regenerable).
* `plotting/data/<ds>[_mean][_<sampler>]/*.dat` + `meta.gp`; `figures/<same tag>/fig_*.png`:
  `fig_umap_latent_spaces`, `fig_dataset_composition`, `fig_dataset_information`,
  `fig_distance_analysis`.

## Provenance of the Si study

The original single-dataset Si pipeline (per-descriptor `compute_and_prune.py`
scripts, `clustering/`, `pruning/`, Si-only gnuplot scripts) is in the git history
up to commit `5075c14`. Its outputs (cuts used for the model fits, descriptor
caches, figures) are kept untracked in `legacy_si/` on the original machine.
