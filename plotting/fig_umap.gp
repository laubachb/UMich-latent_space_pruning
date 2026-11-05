# UMAP embeddings of the structure-level latent spaces, coloured by category,
# on a grid of 3 columns with the legend in a column on the right.
# Data: data/<DS>/umap_<descriptor>.dat   Output: ../figures/<DS>/fig_umap_latent_spaces.png

reset
load "style.gp"
NCOL = 3
NROW = int((NDESC + NCOL - 1) / NCOL)
W_IN = 7.2; H_IN = 2.0 * NROW + 0.3
set terminal pdfcairo enhanced color font FONT.",10" size W_IN, H_IN
OUT = OUTDIR."/fig_umap_latent_spaces"
set output OUT.".pdf"

L0 = 0.045; R0 = 0.78; CGAP = 0.02; KEYL = 0.80
B0 = 0.07; T0 = 0.985; RGAP = 0.03
W = (R0 - L0 - (NCOL - 1)*CGAP) / NCOL
H = (T0 - B0 - (NROW - 1)*RGAP) / NROW
COLX(c) = L0 + (c - 1)*(W + CGAP)
ROWB(r) = T0 - r*H - (r - 1)*RGAP

set multiplot
unset key
unset xtics
unset ytics
set border 15 lw 0.8
set offsets graph 0.06, graph 0.06, graph 0.18, graph 0.06

do for [p=1:NDESC] {
    r = int((p - 1) / NCOL) + 1
    c = p - (r - 1)*NCOL
    eval PANEL(COLX(c), COLX(c) + W, ROWB(r), ROWB(r) + H)
    if (r == NROW) { set xlabel "UMAP 1" offset 0,0.25 } else { unset xlabel }
    if (c == 1)    { set ylabel "UMAP 2" offset 0.9,0 } else { unset ylabel }
    f = DATA(sprintf("umap_%s.dat", DESC[p]))
    set label 1 sprintf("{/:Bold (%c)} %s", 96 + p, DESCNAME[p]) \
        at graph 0.03, graph 0.93 left front font FONT.",9"
    plot for [i=0:NCAT-1] f u 1:($3 == i ? $2 : NaN) w p pt 7 ps 0.5 lc rgb CATCOL[i+1], \
         f u 1:2 w p pt 6 ps 0.5 lw 0.12 lc rgb "#555555"
}

eval KEYSTRIP(B0, T0)
eval PANEL(KEYL, 1.0, B0, T0)
set key vertical maxrows NCAT left center reverse Left samplen 0.6 spacing 1.1 \
    font FONT.",9" title sprintf("%s category", ELEMENT) left
plot for [k=1:NCAT] 1/0 w p pt 7 ps 0.8 lc rgb CATCOL[CATKEY[k]] title CATNAME[CATKEY[k]]
unset multiplot
eval FINISH
