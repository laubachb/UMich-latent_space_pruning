"""Analyses of the latent spaces and of the training sets they select.

    composition      category composition of the cuts vs retention (pooled over replicates)
    quests           QUESTS information of the cuts (D/D_full, coverage, H/H_full)
    distance         category-resolved interpoint distances and FPS enrichment
    embeddings       UMAP / PCA coordinates of each latent space
    dimensionality   participation ratio of each latent space
    crossspace       distance-rank agreement and kNN overlap between spaces; cut overlap vs retention
    labels           label-space coverage of the cuts (W1/sigma of E/atom, |F|, stress)
    export           writes plotting/data/<dataset>/ for the gnuplot figures

Each `run(dataset)` writes CSV tables to results/<dataset>/analysis/.
"""
