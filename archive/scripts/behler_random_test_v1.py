from __future__ import annotations
from monty.serialization import loadfn, dumpfn
import numpy as np
import os
import random


def structure_level_pruning(data, pruning_ratio, aggregation_method='none',
                          selection_method='random', random_seed=None):
    """
    Perform structure-level pruning using random sampling.

    Args:
        data: list of structure dictionaries
        pruning_ratio: fraction of structures to keep (0.1 = keep 10%)
        aggregation_method: not used for random sampling
        selection_method: 'random' for random sampling
        random_seed: random seed for reproducible sampling

    Returns:
        pruned_data: selected subset of structures
        selection_info: metadata about the selection process
    """

    print(f"Starting structure-level pruning with random sampling...")
    print(f"Target: keep {pruning_ratio:.1%} of {len(data)} structures")

    # Select structures based on chosen method
    n_select = max(1, int(len(data) * pruning_ratio))

    print(f"Using random selection for {n_select} structures...")
    if random_seed is not None:
        np.random.seed(random_seed)
        print(f"Using random seed: {random_seed}")
    selected_indices = np.random.choice(len(data), n_select, replace=False)

    # Create pruned dataset
    pruned_data = [data[i] for i in selected_indices]

    # Create selection metadata
    selection_info = {
        'method': selection_method,
        'aggregation': aggregation_method,
        'n_original': len(data),
        'n_selected': len(pruned_data),
        'pruning_ratio': pruning_ratio,
        'selected_indices': selected_indices.tolist(),
        'random_seed': random_seed
    }

    return pruned_data, selection_info

def main():
    """Main function to demonstrate random structure sampling with replicates"""

    # Load data
    print("Loading data...")
    data = loadfn("../data.json")
    print(f"Loaded {len(data)} structures")

    # Define pruning parameters
    pruning_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    selection_method = 'random'
    n_replicates = 10

    # Create output directory
    output_dir = "random_replicates_structure_pruning"
    os.makedirs(output_dir, exist_ok=True)

    # Generate random seeds for replicates
    random.seed(42)  # For reproducible replicate generation
    replicate_seeds = [random.randint(0, 2**32-1) for _ in range(n_replicates)]

    # Process each ratio with replicates
    for ratio in pruning_ratios:
        for replicate_idx in range(n_replicates):
            print(f"\n{'='*60}")
            print(f"Processing: random sampling, {ratio:.0%} retention, replicate {replicate_idx+1}/{n_replicates}")
            print(f"{'='*60}")

            # Perform random sampling
            pruned_data, selection_info = structure_level_pruning(
                data,
                pruning_ratio=ratio,
                aggregation_method='none',  # Not used for random sampling
                selection_method=selection_method,
                random_seed=replicate_seeds[replicate_idx]
            )

            # Save results
            percentage = ratio * 100
            data_filename = f"{output_dir}/si_structures_random_{percentage:.0f}percent_replicate{replicate_idx+1:02d}.json"
            info_filename = f"{output_dir}/si_structures_random_{percentage:.0f}percent_replicate{replicate_idx+1:02d}_info.json"

            dumpfn(pruned_data, data_filename, indent=2)
            dumpfn(selection_info, info_filename, indent=2)

            print(f"Saved {len(pruned_data)} structures to {data_filename}")
            print(f"Selection info saved to {info_filename}")
            print(f"Random seed used: {replicate_seeds[replicate_idx]}")

if __name__ == "__main__":
    main()