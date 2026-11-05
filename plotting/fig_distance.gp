# Distance analysis, one column per latent space:
#   top row    : max-normalised interpoint-distance distribution, stacked by the
#                category that owns each directed distance (stack = full distribution)
#   bottom row : each category's mean normalised distance to the other frames vs its
#                FPS enrichment at <= 10 % retention, with the Spearman rho across categories
# Data: data/<DS>/distance_hist_<desc>.dat, distance_selection_<desc>.dat, distance_rho.gp
# Output: ../figures/<DS>/fig_distance_analysis.png

reset
load "style.gp"
load DATA("distance_rho.gp")
W_IN = 1.25 * NDESC + 0.6; H_IN = 4
set terminal pdfcairo enhanced color font FONT.",9" size W_IN, H_IN
OUT = OUTDIR."/fig_distance_analysis"
set output OUT.".pdf"

L0 = 0.5 / W_IN; R0 = 0.99; CGAP = 0.03
W = (R0 - L0 - (NDESC - 1)*CGAP) / NDESC
COLX(c) = L0 + (c - 1)*(W + CGAP)
TOP_B = 0.625; TOP_T = 0.93
BOT_B = 0.235; BOT_T = 0.50

set multiplot
unset key
set border 3 lw 0.8 front
set tics front
set mxtics 2
set mytics 2
set label 10 "Distance / max distance" at screen (L0 + R0)/2, screen TOP_B - 0.072 center
set label 11 "Mean distance to other frames" at screen (L0 + R0)/2, screen BOT_B - 0.075 center
set label 12 sprintf("{/:Bold %s}", ELEMENT) at screen 0.99, screen 0.985 right font FONT.",10"

# ---- top row: stacked distance distributions --------------------------------
set xrange [0:1]
set xtics ("0" 0, "0.5" 0.5, "1" 1) offset 0,0.3
set yrange [0:5]
set ytics 0,1,5
set style fill solid 1.0 noborder
do for [c=1:NDESC] {
    eval PANEL(COLX(c), COLX(c) + W, TOP_B, TOP_T)
    set title DESCNAME[c] offset 0,-0.7 font FONT.",8"
    if (c == 1) { set format y "%g"; set ylabel "Density" offset 0.4,0 } \
    else        { set format y ""; unset ylabel }
    set label 1 sprintf("{/:Bold (%c)}", 96 + c) at graph 0.05, graph 0.93 left front
    f = DATA(sprintf("distance_hist_%s.dat", DESC[c]))
    plot for [i=NCAT:1:-1] f u 1:(column(i+1)) w fillsteps fc rgb CATCOL[i], \
         f u 1:(column(NCAT+1)) w steps lc rgb "#333333" lw 0.6
}
unset title

# ---- bottom row: isolation vs FPS enrichment ---------------------------------
set xrange [0:0.8]
set xtics ("0.2" 0.2, "0.4" 0.4, "0.6" 0.6) offset 0,0.3
set yrange [0:*]
set ytics auto
set format y "%g"
set arrow 1 from graph 0, first 1 to graph 1, first 1 nohead dt 3 lw 0.8 lc rgb "#000000" back
do for [c=1:NDESC] {
    eval PANEL(COLX(c), COLX(c) + W, BOT_B, BOT_T)
    if (c == 1) { set ylabel "FPS enrichment" offset 0.4,0 } else { unset ylabel }
    set label 1 sprintf("{/:Bold (%c)}", 96 + NDESC + c) at graph 0.05, graph 0.91 left front
    set label 2 sprintf("ρ = %.2f", RHO[c]) at graph 0.95, graph 0.91 right front
    f = DATA(sprintf("distance_selection_%s.dat", DESC[c]))
    plot for [i=0:NCAT-1] f u 2:($1 == i ? $4 : NaN) w p pt 7 ps 0.65 lc rgb CATCOL[i+1], \
         f u 2:4 w p pt 6 ps 0.65 lw 0.15 lc rgb "#555555"
}
unset label 2
unset arrow 1

eval KEYSTRIP(0.025, 0.135)
eval PANEL(L0, R0, 0.025, 0.135)
set key font FONT.",8" maxrows 2 samplen 1.2 width -1
set style fill solid 1.0 noborder
plot for [k=1:NCAT] 1/0 w boxes fc rgb CATCOL[CATKEY[k]] title CATNAME[CATKEY[k]]
unset multiplot
eval FINISH
