# Cu: categories and methods for the gnuplot figures (written by lsp.analysis.export)
DATASET = "Cu"
ELEMENT = "Cu"
SAMPLER = "kmedoids"
NCAT = 7
array CATNAME[NCAT] = ["Surface", "Vacancy 300 K", "Vacancy 3000 K", "Bulk 3000 K", "Bulk 1000 K", "Bulk 300 K", "Strained"]
array CATCOL[NCAT] = ["#E054A6", "#ABD1E3", "#1F4E8C", "#D0662A", "#FCA95E", "#FFF3DF", "#45A05A"]
array CATKEY[NCAT] = [7, 1, 2, 3, 6, 5, 4]
NMETH = 8
NDESC = 7
array METH[NMETH] = ["random", "kmedoids-soap_mean", "kmedoids-behler_mean", "kmedoids-bispectrum_mean", "kmedoids-mtp_mean", "kmedoids-chimes", "kmedoids-chimes_fingerprint", "kmedoids-euler"]
array METHNAME[NMETH] = ["Random", "SOAP (per-atom mean)", "Behler–Parrinello (per-atom mean)", "Bispectrum (per-atom mean)", "MTP (per-atom mean)", "ChIMES", "ChIMES fingerprint", "Euler characteristic"]
array METHCOL[NMETH] = ["#555555", "#3A86FF", "#06A77D", "#F77F00", "#8338EC", "#E63946", "#A4161A", "#7F7F00"]
array METHDT[NMETH] = [2, 1, 1, 1, 1, 1, 1, 1]
array METHPT[NMETH] = [7, 5, 9, 11, 13, 15, 3, 1]
array DESC[NDESC] = ["soap_mean", "behler_mean", "bispectrum_mean", "mtp_mean", "chimes", "chimes_fingerprint", "euler"]
array DESCNAME[NDESC] = ["SOAP (per-atom mean)", "Behler–Parrinello (per-atom mean)", "Bispectrum (per-atom mean)", "MTP (per-atom mean)", "ChIMES", "ChIMES fingerprint", "Euler characteristic"]
array DESCSHORT[NDESC] = ["SOAP", "BP", "Bisp.", "MTP", "ChIMES", "ChIMES FP", "Euler"]
