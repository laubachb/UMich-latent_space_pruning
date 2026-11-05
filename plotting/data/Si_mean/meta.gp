# Si: categories and methods for the gnuplot figures (written by lsp.analysis.export)
DATASET = "Si"
ELEMENT = "Si"
SAMPLER = "fps"
NCAT = 9
array CATNAME[NCAT] = ["Surface", "Vacancy 300 K", "Vacancy 3374 K", "Bulk 3374 K", "Bulk 2530 K", "Bulk 1518 K", "Bulk 843 K", "Bulk 300 K", "Strained"]
array CATCOL[NCAT] = ["#E054A6", "#ABD1E3", "#1F4E8C", "#D0662A", "#F0883A", "#FCA95E", "#FEE5C4", "#FFF3DF", "#45A05A"]
array CATKEY[NCAT] = [9, 1, 2, 3, 8, 7, 6, 5, 4]
NMETH = 7
NDESC = 6
array METH[NMETH] = ["random", "fps-soap_mean", "fps-behler_mean", "fps-bispectrum_mean", "fps-mtp_mean", "fps-chimes", "fps-chimes_fingerprint"]
array METHNAME[NMETH] = ["Random", "SOAP (per-atom mean)", "Behler–Parrinello (per-atom mean)", "Bispectrum (per-atom mean)", "MTP (per-atom mean)", "ChIMES", "ChIMES fingerprint"]
array METHCOL[NMETH] = ["#555555", "#3A86FF", "#06A77D", "#F77F00", "#8338EC", "#E63946", "#A4161A"]
array METHDT[NMETH] = [2, 1, 1, 1, 1, 1, 1]
array METHPT[NMETH] = [7, 5, 9, 11, 13, 15, 3]
array DESC[NDESC] = ["soap_mean", "behler_mean", "bispectrum_mean", "mtp_mean", "chimes", "chimes_fingerprint"]
array DESCNAME[NDESC] = ["SOAP (per-atom mean)", "Behler–Parrinello (per-atom mean)", "Bispectrum (per-atom mean)", "MTP (per-atom mean)", "ChIMES", "ChIMES fingerprint"]
