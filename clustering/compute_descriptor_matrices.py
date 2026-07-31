"""
Compute and cache the structure-level descriptor matrix for every descriptor
space, plus a shared per-frame metadata table.

Descriptors whose dependencies are unavailable (dscribe, maml/LAMMPS, a ChIMES
pkl, ...) are skipped with a warning so the analysis still runs on whatever is
installed. In a bare environment only the topological `euler` descriptor (numpy
+ ase) is guaranteed to succeed.

Run from this directory:
    python compute_descriptor_matrices.py            # all available descriptors
    python compute_descriptor_matrices.py euler soap # a subset

Outputs (clustering/cache/):
    <name>_descriptors.npy   raw (n_frames, n_features) matrix per descriptor
    metadata.csv             per-frame group / category / T / E-per-atom / ...
"""

from __future__ import annotations

import sys
import pandas as pd

import common


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    requested = argv if argv else list(common.DESCRIPTOR_BUILDERS)

    data = common.load_frames()

    # Metadata (cheap, always succeeds) --------------------------------------
    meta = common.extract_metadata(data)
    meta_df = pd.DataFrame(meta)
    meta_path = common.CACHE_DIR / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"Wrote metadata for {len(meta_df)} frames -> {meta_path}")
    print("  group counts:", dict(meta_df["group"].value_counts()))

    # Descriptor matrices -----------------------------------------------------
    succeeded, skipped = [], []
    for name in requested:
        if name not in common.DESCRIPTOR_BUILDERS:
            print(f"[skip] unknown descriptor '{name}'")
            skipped.append(name)
            continue
        print(f"\n=== Computing '{name}' descriptors ===")
        try:
            matrix = common.DESCRIPTOR_BUILDERS[name](data)
            common.save_descriptor_matrix(name, matrix)
            succeeded.append(name)
        except Exception as exc:  # missing lib / LAMMPS / pkl / alignment issue
            print(f"[skip] '{name}' unavailable: {type(exc).__name__}: {exc}")
            skipped.append(name)

    print("\n" + "=" * 60)
    print(f"Cached descriptors: {succeeded or 'none'}")
    if skipped:
        print(f"Skipped:            {skipped}")
    print("Next: python distance_analysis.py && python latent_embeddings.py "
          "&& python cluster_motifs.py")


if __name__ == "__main__":
    main()
