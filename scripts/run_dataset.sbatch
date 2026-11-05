#!/bin/bash
# Run the full pipeline for one dataset on a pdebug node.
#   sbatch --export=DS=Cu scripts/run_dataset.sbatch
# Logs: results/<DS>/logs/pipeline_<jobid>.log
#SBATCH -p pdebug
#SBATCH -A pls2
#SBATCH -N 1
#SBATCH --ntasks-per-node=112
#SBATCH -t 01:00:00
#SBATCH -J lsp
#SBATCH -o /p/lustre2/laubach2/UMich-latent_space_pruning/results/slurm_%j.log
set -u
: "${DS:?set DS=<dataset> with --export}"
R=/p/lustre2/laubach2/UMich-latent_space_pruning
cd "$R"
mkdir -p results/$DS/logs
module load intel-classic/2021.6.0 mvapich2/2.3.7 >/dev/null 2>&1   # libmpi for the LAMMPS builds
export MPLBACKEND=Agg
export TMPDIR=/tmp/$USER/lsp_$SLURM_JOB_ID
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=112
echo "== $DS on $(hostname) $(date)"
.venv/bin/python scripts/run_pipeline.py "$DS" --skip-failures ${STAGES:+--stage $STAGES} ${EXTRA:-} 2>&1 \
    | grep -v "pkg_resources\|UserWarning" | tee results/$DS/logs/pipeline_$SLURM_JOB_ID.log
rm -rf "$TMPDIR"
echo "== done $(date)"
