# Cross-space comparison, three panels in a row:
#   (a) distance-rank agreement between latent spaces (Spearman of pairwise frame distances)
#   (b) local neighbourhood overlap (mean Jaccard of each frame's k nearest neighbours)
#   (c) overlap of the selected cuts vs retention: two different spaces (mean, band = min-max
#       over space pairs), the same space with another seed, and two random cuts
# Data: data/<DS>/crossspace_matrix.dat, cut_overlap.dat, crossspace.gp
# Output: ../figures/<DS>/fig_crossspace.png

reset
load "style.gp"
load DATA("crossspace.gp")
W_IN = 7.2; H_IN = 2.95
set terminal pdfcairo enhanced color font FONT.",9" size W_IN, H_IN
OUT = OUTDIR."/fig_crossspace"
set output OUT.".pdf"

SIDE = 1.95                                   # heatmap side (inches)
B0 = 0.62 / H_IN; T0 = B0 + SIDE / H_IN
XA = 0.66 / W_IN; XB = XA + (SIDE + 0.1) / W_IN
XC0 = XB + (SIDE + 0.55) / W_IN; XC1 = 0.985
set multiplot
set label 12 sprintf("{/:Bold %s}", ELEMENT) at screen 0.99, screen 1 - 0.05/H_IN right font FONT.",10"

# ---- heatmaps ------------------------------------------------------------------
set palette defined (0 "#f4f6fb", 0.5 "#8fb3d9", 0.75 "#3a6fb0", 1 "#0d2f5e")
set cbrange [0:1]
unset colorbox
set xrange [-0.5:NDESC-0.5]
set yrange [NDESC-0.5:-0.5]
set size ratio -1
set border 15 lw 0.6
set tics scale 0
set xtics rotate by 45 right offset 0.4,0 font FONT.",8"
set ytics font FONT.",8"
set for [i=1:NDESC] xtics add (DESCSHORT[i] i-1)
set for [i=1:NDESC] ytics add (DESCSHORT[i] i-1)
f = DATA("crossspace_matrix.dat")
array HCOL[2] = [3, 4]
array HTITLE[2] = ["Distance-rank agreement", sprintf("k-NN overlap (k = 10; random %.2f)", KNN_RANDOM)]
do for [h=1:2] {
    if (h == 1) { eval PANEL(XA, XA + SIDE/W_IN, B0, T0) } \
    else        { eval PANEL(XB, XB + SIDE/W_IN, B0, T0); unset ytics }
    set title HTITLE[h] offset 0,-0.6 font FONT.",9"
    set label 1 sprintf("{/:Bold (%c)}", 96 + h) at graph -0.02, graph 1.09 right front font FONT.",9"
    c = HCOL[h]
    plot f u 1:2:(column(c)) w image notitle, \
         f u 1:2:(column(c) <  0.62 ? sprintf("%.2f", column(c)) : "") w labels font FONT.",6" tc rgb "#1d2129" notitle, \
         f u 1:2:(column(c) >= 0.62 ? sprintf("%.2f", column(c)) : "") w labels font FONT.",6" tc rgb "#ffffff" notitle
}
unset label 1
unset title
set size noratio
set tics scale 0.6
set border 3 lw 0.8
set xtics norotate offset 0,0.3 font FONT.",9"
set ytics font FONT.",9"
set format y "%.1f"

# ---- cut overlap vs retention --------------------------------------------------
eval PANEL(XC0, XC1, B0, T0)
set xtics ()
set ytics ()
set logscale x
set xrange [0.85:100]
set xtics ("1" 1, "10" 10, "90" 90) offset 0,0.3
set mxtics 10
set yrange [0:1]
set ytics 0, 0.2, 1
set mytics 2
set xlabel "Retention (%)" offset 0,0.6
set ylabel "Cut overlap (Jaccard)" offset 1.2,0
set label 1 "{/:Bold (c)}" at graph 0.04, graph 0.93 left front font FONT.",9"
set key top left at graph 0.02, graph 0.86 reverse Left samplen 1.6 spacing 0.95 font FONT.",7.5"
set style fill transparent solid 0.25 noborder
o = DATA("cut_overlap.dat")
# columns: 1 retention, 2 between-space mean, 3 min, 4 max over space pairs, 5 same space, 6 random
plot o u 1:3:4 w filledcurves lc rgb "#B5421B" notitle, \
     o u 1:2 w lp lw 1.4 pt 7 ps 0.35 lc rgb "#B5421B" title sprintf("%s, different spaces", SAMPLER), \
     o u 1:5 w lp lw 1.2 dt 2 pt 6 ps 0.35 lc rgb "#3A6FB0" title "same space, other seed", \
     o u 1:6 w l lw 1.0 dt 3 lc rgb "#555555" title "two random cuts"
unset multiplot
eval FINISH
