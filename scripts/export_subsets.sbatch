#!/bin/bash
# Build the training-subset archive (all datasets, 100 replicates x 7 retentions) on a pdebug node.
#   sbatch scripts/export_subsets.sbatch
#SBATCH -p pdebug
#SBATCH -A pls2
#SBATCH -N 1
#SBATCH --ntasks-per-node=112
#SBATCH -t 01:00:00
#SBATCH -J lsp_export
#SBATCH -o /p/lustre2/laubach2/UMich-latent_space_pruning/results/slurm_export_%j.log
set -u
R=/p/lustre2/laubach2/UMich-latent_space_pruning
cd "$R"
export OMP_NUM_THREADS=1 LSP_NPROC=100
OUT=${OUT:-/p/lustre2/laubach2/UMich-latent_space_pruning/results/lsp_training_subsets_$(date +%Y%m%d).tar.gz}
echo "== export on $(hostname) $(date) -> $OUT"
.venv/bin/python scripts/export_training_subsets.py ${DATASETS:-} --out "$OUT" 2>&1 \
    | grep -v "pkg_resources\|UserWarning\|ConvergenceWarning\|warn(\|fit_method"
ls -la "$OUT"
echo "== done $(date)"
