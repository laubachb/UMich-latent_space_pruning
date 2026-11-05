#!/bin/bash
# Render the publication figures of one or more datasets:
#   cd plotting && ./make_figures.sh Si Cu Ge Li Mo Ni Si_mean
# Reads data/<DS>/ (lsp.analysis.export) and writes ../figures/<DS>/fig_*.png (300 dpi).
set -e
cd "$(dirname "$0")"
for DS in "$@"; do
    [ -f "data/$DS/meta.gp" ] || { echo "no data/$DS/meta.gp (run the export stage first)"; continue; }
    for f in fig_umap.gp fig_composition.gp fig_information.gp fig_distance.gp; do
        gnuplot -e "DS='$DS'" "$f" && echo "$DS: $f -> ../figures/$DS/"
    done
done
