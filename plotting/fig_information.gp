# QUESTS information content of the pruned training sets vs retention, three panels:
# (a) diversity D(S)/D(full), (b) coverage of the full dataset, (c) entropy H(S)/H(full);
# mean ± s.d. over the replicates; legend strip underneath.
# Data: data/<DS>/information_<method>.dat   Output: ../figures/<DS>/fig_dataset_information.png

reset
load "style.gp"
W_IN = 7.2; H_IN = 2.7
set terminal pdfcairo enhanced color font FONT.",10" size W_IN, H_IN
OUT = OUTDIR."/fig_dataset_information"
set output OUT.".pdf"

L0 = 0.075; R0 = 0.99; CGAP = 0.08; B0 = 0.33; T0 = 0.95
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

array YCOL[3]  = [2, 4, 6]
array YLAB[3]  = ["{/:Italic D}({/:Italic S}) / {/:Italic D}(full)", \
                  "Coverage of full dataset", \
                  "{/:Italic H}({/:Italic S}) / {/:Italic H}(full)"]
array YMAX[3]  = [1.15, 1.15, 1.6]
array YSTEP[3] = [0.2, 0.2, 0.25]
array YFMT[3]  = ["%.1f", "%.1f", "%.2f"]

do for [p=1:3] {
    eval PANEL(COLX(p), COLX(p) + W, B0, T0)
    set yrange [0:YMAX[p]]
    set ytics YSTEP[p]
    set format y YFMT[p]
    set ylabel YLAB[p] offset 0.6,0
    set label 1 sprintf("{/:Bold (%c)}", 96 + p) at graph 0.04, graph 0.93 left front font FONT.",10"
    set arrow 1 from graph 0, first 1 to graph 1, first 1 nohead dt 3 lw 0.8 lc rgb "#000000" back
    c = YCOL[p]
    plot for [m=1:NMETH] DATA(sprintf("information_%s.dat", METH[m])) \
             u 1:(column(c)):(column(c+1)) w yerrorbars \
             lc rgb METHCOL[m] lw 0.8 pt METHPT[m] ps 0.4 notitle, \
         for [m=1:NMETH] DATA(sprintf("information_%s.dat", METH[m])) \
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
