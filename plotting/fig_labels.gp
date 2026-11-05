# Label-space coverage of the pruned training sets vs retention, three panels:
# 1-Wasserstein distance between the label distribution of a cut and that of the full set,
# divided by the full set's standard deviation, for (a) energy per atom (per frame),
# (b) force magnitude (per atom), (c) mean normal stress (per frame); 0 = identical
# distributions. Mean ± s.d. over the replicates; legend strip underneath.
# Data: data/<DS>/labels_<method>.dat   Output: ../figures/<DS>/fig_label_coverage.png

reset
load "style.gp"
W_IN = 7.2; H_IN = 2.7
set terminal pdfcairo enhanced color font FONT.",10" size W_IN, H_IN
OUT = OUTDIR."/fig_label_coverage"
set output OUT.".pdf"

L0 = 0.075; R0 = 0.99; CGAP = 0.075; B0 = 0.33; T0 = 0.95
W = (R0 - L0 - 2*CGAP) / 3.0
COLX(p) = L0 + (p - 1)*(W + CGAP)

set multiplot
unset key
set logscale x
set xrange [0.85:115]
set xtics ("1" 1, "10" 10, "100" 100) offset 0,0.3
set mxtics 10
set ytics nomirror
set mytics 2
set bars 0.6
set border 3 lw 0.8
set label 10 "Retention (%)" at screen (L0 + R0)/2, screen 0.23 center
set label 12 sprintf("{/:Bold %s}", ELEMENT) at screen 0.99, screen 0.985 right font FONT.",10"

array YCOL[3] = [2, 4, 6]
array YLAB[3] = ["Energy per atom", "Force magnitude", "Mean normal stress"]
do for [p=1:3] {
    eval PANEL(COLX(p), COLX(p) + W, B0, T0)
    set yrange [0:*]
    set format y "%.2f"
    set ylabel sprintf("{/:Italic W}_1 / {/Symbol s}_{full}  (%s)", YLAB[p]) offset 0.8,0 font FONT.",9"
    set label 1 sprintf("{/:Bold (%c)}", 96 + p) at graph 0.92, graph 0.93 right front font FONT.",10"
    c = YCOL[p]
    plot for [m=1:NMETH] DATA(sprintf("labels_%s.dat", METH[m])) \
             u 1:(column(c)):(column(c+1)) w yerrorbars \
             lc rgb METHCOL[m] lw 0.8 pt METHPT[m] ps 0.4 notitle, \
         for [m=1:NMETH] DATA(sprintf("labels_%s.dat", METH[m])) \
             u 1:(column(c)) w lp lc rgb METHCOL[m] lw 1.2 dt METHDT[m] \
             pt METHPT[m] ps 0.4 notitle
}

eval KEYSTRIP(0.01, 0.15)
eval PANEL(L0, R0, 0.01, 0.15)
set key font FONT.",8" samplen 1.6 width -2 maxrows 2
plot for [m=1:NMETH] 1/0 w lp lc rgb METHCOL[m] lw 1.2 dt METHDT[m] pt METHPT[m] ps 0.5 \
         title METHNAME[m]
unset multiplot
eval FINISH
