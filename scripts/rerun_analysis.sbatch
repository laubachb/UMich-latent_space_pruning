#!/bin/bash
# Redo selection + analysis + plot-data export for several datasets on one pdebug node
# (descriptors are cached), then check the QUESTS scoring against reference values.
#   sbatch --export=DATASETS="Si Ge Li" scripts/rerun_analysis.sbatch
#SBATCH -p pdebug
#SBATCH -A pls2
#SBATCH -N 1
#SBATCH --ntasks-per-node=112
#SBATCH -t 01:00:00
#SBATCH -J lsp_rerun
#SBATCH -o /p/lustre2/laubach2/UMich-latent_space_pruning/results/slurm_rerun_%j.log
set -u
: "${DATASETS:?set DATASETS=\"Si Ge ...\" with --export}"
R=/p/lustre2/laubach2/UMich-latent_space_pruning
cd "$R"
export MPLBACKEND=Agg OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=112 LSP_NPROC=100
for DS in $DATASETS; do
    echo "== $DS on $(hostname) $(date)"
    .venv/bin/python scripts/run_pipeline.py "$DS" --skip-failures --stage ${STAGES:-selection analysis export} 2>&1 \
        | grep -v "pkg_resources\|UserWarning\|ConvergenceWarning\|warn(\|fit_method\|^  "
    REF=results/reference_10rep/quests_$DS.csv
    [ -f "$REF" ] && .venv/bin/python tests/test_quests_frame_kernel.py "$DS" "$REF" 2>&1 | grep -v "pkg_resources\|UserWarning"
done
echo "== done $(date)"
