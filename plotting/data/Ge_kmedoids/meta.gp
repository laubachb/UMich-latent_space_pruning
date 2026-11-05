# Ge: categories and methods for the gnuplot figures (written by lsp.analysis.export)
DATASET = "Ge"
ELEMENT = "Ge"
SAMPLER = "kmedoids"
NCAT = 9
array CATNAME[NCAT] = ["Surface", "Vacancy 300 K", "Vacancy 2422 K", "Bulk 2422 K", "Bulk 1816 K", "Bulk 1089 K", "Bulk 605 K", "Bulk 300 K", "Strained"]
array CATCOL[NCAT] = ["#E054A6", "#ABD1E3", "#1F4E8C", "#D0662A", "#F0883A", "#FCA95E", "#FEE5C4", "#FFF3DF", "#45A05A"]
array CATKEY[NCAT] = [9, 1, 2, 3, 8, 7, 6, 5, 4]
NMETH = 8
NDESC = 7
array METH[NMETH] = ["random", "kmedoids-soap", "kmedoids-behler", "kmedoids-bispectrum", "kmedoids-mtp", "kmedoids-chimes", "kmedoids-chimes_fingerprint", "kmedoids-euler"]
array METHNAME[NMETH] = ["Random", "SOAP", "Behler–Parrinello", "Bispectrum", "MTP", "ChIMES", "ChIMES fingerprint", "Euler characteristic"]
array METHCOL[NMETH] = ["#555555", "#3A86FF", "#06A77D", "#F77F00", "#8338EC", "#E63946", "#A4161A", "#7F7F00"]
array METHDT[NMETH] = [2, 1, 1, 1, 1, 1, 1, 1]
array METHPT[NMETH] = [7, 5, 9, 11, 13, 15, 3, 1]
array DESC[NDESC] = ["soap", "behler", "bispectrum", "mtp", "chimes", "chimes_fingerprint", "euler"]
array DESCNAME[NDESC] = ["SOAP", "Behler–Parrinello", "Bispectrum", "MTP", "ChIMES", "ChIMES fingerprint", "Euler characteristic"]
array DESCSHORT[NDESC] = ["SOAP", "BP", "Bisp.", "MTP", "ChIMES", "ChIMES FP", "Euler"]
