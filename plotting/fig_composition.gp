# Category composition of the pruned training sets vs retention, one panel per
# selection method (random baseline first) on a grid of 4 columns. The legend takes
# the free cell after the last panel; when the grid is full it goes in a two-row
# strip under the panels instead. Pooled over the replicates.
# Data: data/<DS>/composition_<method>.dat   Output: ../figures/<DS>/fig_dataset_composition.png

reset
load "style.gp"
NCOL = 4
STRIP = (NMETH % NCOL == 0) ? 1 : 0                     # 1: no free cell -> legend strip
NROW = int((NMETH + 1 - STRIP + NCOL - 1) / NCOL)
# geometry in inches
PANEL_H = 1.75; ROW_GAP = 0.42; TOP_M = 0.22; XLAB_M = 0.42; STRIP_H = 0.55
W_IN = 7.2
H_IN = TOP_M + NROW*PANEL_H + (NROW - 1)*ROW_GAP + XLAB_M + STRIP*STRIP_H
set terminal pdfcairo enhanced color font FONT.",10" size W_IN, H_IN
OUT = OUTDIR."/fig_dataset_composition"
set output OUT.".pdf"

L0 = 0.075; R0 = 0.99; CGAP = 0.03
W = (R0 - L0 - (NCOL - 1)*CGAP) / NCOL
H = PANEL_H / H_IN
RGAP = ROW_GAP / H_IN
T0 = 1 - TOP_M / H_IN
B0 = (XLAB_M + STRIP*STRIP_H) / H_IN
COLX(c) = L0 + (c - 1)*(W + CGAP)
ROWB(r) = T0 - r*H - (r - 1)*RGAP

set multiplot
unset key
set logscale x
set xrange [1:100]
set xtics ("1" 1, "10" 10, "100" 100) offset 0,0.3
set mxtics 10
set yrange [0:100]
set ytics 0,20,100
set mytics 2
set border 3 lw 0.8 front
set tics front
set style fill solid 1.0 noborder
set label 10 "Retention (%)" at screen (L0 + R0)/2, screen (STRIP*STRIP_H + 0.1)/H_IN center
set label 11 "Composition (%)" at screen 0.02, screen (B0 + T0)/2 center rotate by 90
set label 12 sprintf("{/:Bold %s}", ELEMENT) at screen 0.99, screen 1 - 0.06/H_IN right font FONT.",10"

do for [m=1:NMETH] {
    r = int((m - 1) / NCOL) + 1
    c = m - (r - 1)*NCOL
    eval PANEL(COLX(c), COLX(c) + W, ROWB(r), ROWB(r) + H)
    f = DATA(sprintf("composition_%s.dat", METH[m]))
    set title METHNAME[m] offset 0,-0.7 font FONT.",10"
    if (c == 1) { set format y "%g" } else { set format y "" }
    plot for [i=1:NCAT] f u 1:(i == 1 ? 0 : column(i)):(column(i+1)) \
             w filledcurves fc rgb CATCOL[i], \
         for [i=2:NCAT] f u 1:i w l lw 0.8 lc rgb "white"
}

if (STRIP) {
    # two-row legend strip under the panels
    eval KEYSTRIP(0.02/H_IN, (STRIP_H - 0.02)/H_IN)
    eval PANEL(L0, R0, 0.02/H_IN, (STRIP_H - 0.02)/H_IN)
    set key font FONT.",9" maxrows 2 samplen 1.2 width -1
    set style fill solid 1.0 noborder
    plot for [k=1:NCAT] 1/0 w boxes fc rgb CATCOL[CATKEY[k]] title CATNAME[CATKEY[k]]
} else {
    # legend in the next free cell
    m = NMETH + 1
    r = int((m - 1) / NCOL) + 1
    c = m - (r - 1)*NCOL
    eval KEYSTRIP(ROWB(r), ROWB(r) + H)
    eval PANEL(COLX(c), R0, ROWB(r) - 0.02, ROWB(r) + H)
    set key vertical maxrows NCAT left center reverse Left samplen 1.2 spacing 1.0 \
        font FONT.",9" title "Category" left
    set style fill solid 1.0 noborder
    plot for [k=1:NCAT] 1/0 w boxes fc rgb CATCOL[CATKEY[k]] title CATNAME[CATKEY[k]]
}
unset multiplot
eval FINISH
