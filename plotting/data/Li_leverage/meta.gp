# Li: categories and methods for the gnuplot figures (written by lsp.analysis.export)
DATASET = "Li"
ELEMENT = "Li"
SAMPLER = "leverage"
NCAT = 8
array CATNAME[NCAT] = ["Surface", "Vacancy 300 K", "Vacancy 907 K", "Bulk 907 K", "Bulk 680 K", "Bulk 408 K", "Bulk 300 K", "Strained"]
array CATCOL[NCAT] = ["#E054A6", "#ABD1E3", "#1F4E8C", "#D0662A", "#FCA95E", "#FDC987", "#FFF3DF", "#45A05A"]
array CATKEY[NCAT] = [8, 1, 2, 3, 7, 6, 5, 4]
NMETH = 8
NDESC = 7
array METH[NMETH] = ["random", "leverage-soap", "leverage-behler", "leverage-bispectrum", "leverage-mtp", "leverage-chimes", "leverage-chimes_fingerprint", "leverage-euler"]
array METHNAME[NMETH] = ["Random", "SOAP", "Behler–Parrinello", "Bispectrum", "MTP", "ChIMES", "ChIMES fingerprint", "Euler characteristic"]
array METHCOL[NMETH] = ["#555555", "#3A86FF", "#06A77D", "#F77F00", "#8338EC", "#E63946", "#A4161A", "#7F7F00"]
array METHDT[NMETH] = [2, 1, 1, 1, 1, 1, 1, 1]
array METHPT[NMETH] = [7, 5, 9, 11, 13, 15, 3, 1]
array DESC[NDESC] = ["soap", "behler", "bispectrum", "mtp", "chimes", "chimes_fingerprint", "euler"]
array DESCNAME[NDESC] = ["SOAP", "Behler–Parrinello", "Bispectrum", "MTP", "ChIMES", "ChIMES fingerprint", "Euler characteristic"]
