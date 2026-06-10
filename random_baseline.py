"""
Random-sampling baseline for structure selection.

Generates 10 replicates of random subsets at each pruning ratio, matching
the replicate structure used by the FPS descriptor scripts.  These serve as
the null baseline for comparing descriptor-induced sampling bias.

Run from the project root:
    python random_baseline.py

Output: random_replicates_structure_pruning/ (created in the project root)
"""

from __future__ import annotations
from monty.serialization import loadfn, dumpfn
import numpy as np
import os
import random


def structure_level_pruning(data, pruning_ratio, random_seed=None):
    print(f"Random sampling: keep {pruning_ratio:.1%} of {len(data)} structures")

    n_select = max(1, int(len(data) * pruning_ratio))

    if random_seed is not None:
        np.random.seed(random_seed)
    selected_indices = np.random.choice(len(data), n_select, replace=False)
    pruned_data = [data[i] for i in selected_indices]

    selection_info = {
        'method': 'random',
        'n_original': len(data),
        'n_selected': len(pruned_data),
        'pruning_ratio': pruning_ratio,
        'selected_indices': selected_indices.tolist(),
        'random_seed': random_seed,
    }

    return pruned_data, selection_info


def main():
    print("Loading data...")
    data = loadfn("data.json")
    print(f"Loaded {len(data)} structures")

    pruning_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_replicates = 10

    output_dir = "random_replicates_structure_pruning"
    os.makedirs(output_dir, exist_ok=True)

    random.seed(42)
    replicate_seeds = [random.randint(0, 2**32 - 1) for _ in range(n_replicates)]

    for ratio in pruning_ratios:
        for replicate_idx in range(n_replicates):
            print(f"\n{'='*60}")
            print(f"random {ratio:.0%} retention, replicate {replicate_idx+1}/{n_replicates}")
            print(f"{'='*60}")

            pruned_data, selection_info = structure_level_pruning(
                data, pruning_ratio=ratio, random_seed=replicate_seeds[replicate_idx]
            )

            percentage = ratio * 100
            data_fn = f"{output_dir}/si_structures_random_{percentage:.0f}percent_replicate{replicate_idx+1:02d}.json"
            info_fn = f"{output_dir}/si_structures_random_{percentage:.0f}percent_replicate{replicate_idx+1:02d}_info.json"

            dumpfn(pruned_data, data_fn, indent=2)
            dumpfn(selection_info, info_fn, indent=2)

            print(f"Saved {len(pruned_data)} structures  seed={replicate_seeds[replicate_idx]}")


if __name__ == "__main__":
    main()
