"""lsp — latent-space pruning: descriptor-induced sampling bias across MLIP descriptors.

Package layout
    lsp.paths        repository / results / external-data locations, local tool paths
    lsp.config       per-dataset configuration (configs/<name>.json)
    lsp.data         dataset loading (mlearn JSON), categories, metadata
    lsp.descriptors  structure-level descriptor builders (one module per latent space)
    lsp.selection    FPS rankings, random baseline, cut materialisation
    lsp.analysis     composition / QUESTS information / distance / embedding analyses
                     and the gnuplot data exporter
"""

__all__ = ["paths", "config", "data", "descriptors", "selection", "analysis"]
