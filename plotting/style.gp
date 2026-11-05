# Shared style for the publication figures (loaded by every fig_*.gp).
# Each figure script expects DS (the dataset, e.g. "Si" or "Si_mean") to be set:
#     gnuplot -e 'DS="Cu"' fig_composition.gp
# and loads data/<DS>/meta.gp, which defines the categories (NCAT, CATNAME, CATCOL,
# CATKEY) and the methods (NMETH, METH, METHNAME, METHCOL, METHDT, METHPT, NDESC,
# DESC, DESCNAME) of that dataset (written by lsp.analysis.export).
# Vector PDF via cairo; Nimbus Sans is the URW Helvetica clone.

FONT = "Nimbus Sans"
set encoding utf8
if (!exists("DS")) { DS = "Si" }
load sprintf("data/%s/meta.gp", DS)
DATA(f) = sprintf("data/%s/%s", DS, f)
OUTDIR = sprintf("../figures/%s", DS)
system(sprintf("mkdir -p %s", OUTDIR))

set border lw 0.8
set tics nomirror out scale 0.6
unset grid

# Place the next plot of a (layout-free) multiplot at screen fractions:
#   eval PANEL(left, right, bottom, top)
PANEL(l, r, b, t) = sprintf("set lmargin at screen %g; set rmargin at screen %g; \
set bmargin at screen %g; set tmargin at screen %g", l, r, b, t)

# Start an axis-less strip across [b, t] of the page that only draws a key; follow
# with `plot for [...] 1/0 ... title ...`.
KEYSTRIP(b, t) = PANEL(0, 1, b, t)."; unset border; unset tics; unset xlabel; \
unset ylabel; unset title; unset label; unset arrow; unset logscale; unset object; \
set xrange [0:1]; set yrange [0:1]; set key center center horizontal maxrows 1 \
reverse Left width 0.5 samplen 0.8 font FONT.',9'"

# Figures are delivered as 300 dpi PNGs: each script draws to OUT.".pdf" with the
# cairo PDF terminal, then `eval FINISH` rasterizes it with pdftoppm and removes the PDF.
DPI = 300
FINISH = "unset output; system(sprintf('pdftoppm -png -r %d -singlefile %s.pdf %s && \
convert %s.png -crop %dx%d+0+0 +repage -units PixelsPerInch -density %d %s.png && rm -f %s.pdf', \
DPI, OUT, OUT, OUT, int(W_IN*DPI + 0.5), int(H_IN*DPI + 0.5), DPI, OUT, OUT))"
