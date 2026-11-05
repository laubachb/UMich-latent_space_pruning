# UMAP embeddings of the structure-level latent spaces, coloured by category, on a
# grid of 4 columns. The legend takes the free cell after the last panel; when the
# grid is full it goes in a two-row strip under the panels.
# Data: data/<DS>/umap_<descriptor>.dat   Output: ../figures/<DS>/fig_umap_latent_spaces.png

reset
load "style.gp"
NCOL = 4
STRIP = (NDESC % NCOL == 0) ? 1 : 0
NROW = int((NDESC + 1 - STRIP + NCOL - 1) / NCOL)
PANEL_H = 1.6; ROW_GAP = 0.12; TOP_M = 0.08; XLAB_M = 0.3; STRIP_H = 0.55
W_IN = 7.2
H_IN = TOP_M + NROW*PANEL_H + (NROW - 1)*ROW_GAP + XLAB_M + STRIP*STRIP_H
set terminal pdfcairo enhanced color font FONT.",10" size W_IN, H_IN
OUT = OUTDIR."/fig_umap_latent_spaces"
set output OUT.".pdf"

L0 = 0.045; R0 = 0.99; CGAP = 0.018
W = (R0 - L0 - (NCOL - 1)*CGAP) / NCOL
H = PANEL_H / H_IN
RGAP = ROW_GAP / H_IN
T0 = 1 - TOP_M / H_IN
B0 = (XLAB_M + STRIP*STRIP_H) / H_IN
COLX(c) = L0 + (c - 1)*(W + CGAP)
ROWB(r) = T0 - r*H - (r - 1)*RGAP

set multiplot
unset key
unset xtics
unset ytics
set border 15 lw 0.8
set offsets graph 0.06, graph 0.06, graph 0.18, graph 0.06
set label 12 sprintf("{/:Bold %s}", ELEMENT) at screen 0.99, screen 1 - 0.04/H_IN right font FONT.",9"

do for [p=1:NDESC] {
    r = int((p - 1) / NCOL) + 1
    c = p - (r - 1)*NCOL
    eval PANEL(COLX(c), COLX(c) + W, ROWB(r), ROWB(r) + H)
    if (r == NROW || p + NCOL > NDESC) { set xlabel "UMAP 1" offset 0,0.4 font FONT.",9" } else { unset xlabel }
    if (c == 1) { set ylabel "UMAP 2" offset 1.2,0 font FONT.",9" } else { unset ylabel }
    f = DATA(sprintf("umap_%s.dat", DESC[p]))
    set label 1 sprintf("{/:Bold (%c)} %s", 96 + p, DESCNAME[p]) \
        at graph 0.03, graph 0.93 left front font FONT.",8"
    plot for [i=0:NCAT-1] f u 1:($3 == i ? $2 : NaN) w p pt 7 ps 0.45 lc rgb CATCOL[i+1], \
         f u 1:2 w p pt 6 ps 0.45 lw 0.12 lc rgb "#555555"
}

if (STRIP) {
    eval KEYSTRIP(0.02/H_IN, (STRIP_H - 0.02)/H_IN)
    eval PANEL(L0, R0, 0.02/H_IN, (STRIP_H - 0.02)/H_IN)
    set key font FONT.",9" maxrows 2 samplen 0.8 width -1
    plot for [k=1:NCAT] 1/0 w p pt 7 ps 0.8 lc rgb CATCOL[CATKEY[k]] title CATNAME[CATKEY[k]]
} else {
    m = NDESC + 1
    r = int((m - 1) / NCOL) + 1
    c = m - (r - 1)*NCOL
    eval KEYSTRIP(ROWB(r), ROWB(r) + H)
    eval PANEL(COLX(c), R0, ROWB(r), ROWB(r) + H)
    set key vertical maxrows NCAT left center reverse Left samplen 0.6 spacing 1.0 \
        font FONT.",9" title sprintf("%s category", ELEMENT) left
    plot for [k=1:NCAT] 1/0 w p pt 7 ps 0.8 lc rgb CATCOL[CATKEY[k]] title CATNAME[CATKEY[k]]
}
unset multiplot
eval FINISH
