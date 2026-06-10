from __future__ import annotations
from monty.serialization import loadfn, dumpfn
from dscribe.descriptors import SOAP
from ase import Atoms
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from skmatter.sample_selection import FPS
import os
import random
from maml.describers import BispectrumCoefficients

def pymatgen_to_ase(structure):
    """Convert pymatgen Structure to ASE Atoms"""
    symbols = [site.species_string for site in structure]
    positions = structure.cart_coords
    cell = structure.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

def aggregate_atomic_descriptors(atomic_descriptors, method='mean_std'):
    """
    Aggregate atomic descriptors to create structure-level representation.

    Args:
        atomic_descriptors: 2D array of atomic SOAP descriptors for one structure
        method: aggregation method ('mean', 'mean_std', 'sum', 'histogram')

    Returns:
        structure_descriptor: 1D array representing the entire structure
    """
    if method == 'mean':
        return np.mean(atomic_descriptors, axis=0)

    elif method == 'mean_std':
        # Concatenate mean and standard deviation
        mean_desc = np.mean(atomic_descriptors, axis=0)
        std_desc = np.std(atomic_descriptors, axis=0)
        return np.concatenate([mean_desc, std_desc])

    elif method == 'sum':
        return np.sum(atomic_descriptors, axis=0)

    elif method == 'histogram':
        # Create histogram of descriptor values (simplified version)
        # This is more complex and requires careful binning
        n_bins = 50
        hist_features = []
        for i in range(atomic_descriptors.shape[1]):
            hist, _ = np.histogram(atomic_descriptors[:, i], bins=n_bins, density=True)
            hist_features.extend(hist)
        return np.array(hist_features)

    elif method == 'bag_of_atoms':
        # Sort atomic descriptors and use statistical moments
        sorted_descriptors = np.sort(atomic_descriptors, axis=0)
        percentiles = [10, 25, 50, 75, 90]
        features = []
        for p in percentiles:
            features.extend(np.percentile(sorted_descriptors, p, axis=0))
        return np.array(features)

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def calculate_structure_diversity_scores(structure_descriptors, n_neighbors=5):
    """
    Calculate diversity scores for structures based on local density.
    Higher scores indicate more unique structures.
    """
    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(structure_descriptors)
    distances, _ = nbrs.kneighbors(structure_descriptors)

    # Use mean distance to k nearest neighbors as diversity score
    diversity_scores = np.mean(distances[:, 1:], axis=1)  # Exclude self (distance=0)

    return diversity_scores

def structure_level_pruning(data, pruning_ratio, aggregation_method='mean_std',
                          selection_method='fps', soap_params=None, random_seed=None):
    """
    Perform structure-level pruning using atomic SOAP descriptors.

    Args:
        data: list of structure dictionaries
        pruning_ratio: fraction of structures to keep (0.1 = keep 10%)
        aggregation_method: how to aggregate atomic descriptors
        selection_method: 'fps', 'diversity', or 'random'
        soap_params: dictionary of SOAP parameters
        random_seed: random seed for FPS initialization

    Returns:
        pruned_data: selected subset of structures
        selection_info: metadata about the selection process
    """


    print(f"Starting structure-level pruning with {aggregation_method} aggregation...")
    print(f"Target: keep {pruning_ratio:.1%} of {len(data)} structures")

    # Extract pymatgen structures (BispectrumCoefficients works with pymatgen directly)
    structures = [d["structure"] for d in data]
 
    element_profile = {"Si": {"r": 0.5, "w": 1}}

    # Add LAMMPS to PATH so it can be found automatically
    import os
    os.environ['PATH'] = '/opt/homebrew/Cellar/lammps/20250722-update1/bin:' + os.environ.get('PATH', '')

    # Initialize BispectrumCoefficients (LAMMPS will be auto-detected from PATH)
    describer = BispectrumCoefficients(
        rcutfac=4.9, twojmax=8, element_profile=element_profile,
        quadratic=False, pot_fit=True, include_stress=False
    )

    # Calculate atomic BispectrumCoefficients descriptors and aggregate to structure level
    structure_descriptors = []

    for i, structure in enumerate(structures):
        print(f"Processing structure {i+1}/{len(structures)}", end='\r')

        # Get atomic descriptors for this structure using transform_one
        atomic_descriptors_df = describer.transform_one(structure)

        # Convert DataFrame to numpy array for aggregation
        atomic_descriptors = atomic_descriptors_df.values

        # Aggregate to structure level
        structure_descriptor = aggregate_atomic_descriptors(
            atomic_descriptors, method=aggregation_method
        )
        structure_descriptors.append(structure_descriptor)

    print()  # New line after progress updates
    structure_descriptors = np.array(structure_descriptors)

    # Standardize structure descriptors
    scaler = StandardScaler()
    structure_descriptors_scaled = scaler.fit_transform(structure_descriptors)

    print(f"Structure descriptors shape: {structure_descriptors_scaled.shape}")

    # Select structures based on chosen method
    n_select = max(1, int(len(data) * pruning_ratio))

    if selection_method == 'fps':
        print(f"Using FPS to select {n_select} structures...")
        if random_seed is not None:
            fps = FPS(n_to_select=n_select, random_state=random_seed)
            print(f"Using random seed: {random_seed}")
        else:
            fps = FPS(n_to_select=n_select)
        selected_indices = fps.fit(structure_descriptors_scaled).selected_idx_

    elif selection_method == 'diversity':
        print(f"Using diversity-based selection for {n_select} structures...")
        diversity_scores = calculate_structure_diversity_scores(structure_descriptors_scaled)
        selected_indices = np.argsort(diversity_scores)[-n_select:]  # Most diverse

    elif selection_method == 'random':
        print(f"Using random selection for {n_select} structures...")
        np.random.seed(42)  # For reproducibility
        selected_indices = np.random.choice(len(data), n_select, replace=False)

    else:
        raise ValueError(f"Unknown selection method: {selection_method}")

    # Create pruned dataset
    pruned_data = [data[i] for i in selected_indices]

    # Calculate selection statistics
    if selection_method in ['fps', 'diversity']:
        # Calculate diversity of selected vs original dataset
        all_distances = pairwise_distances(structure_descriptors_scaled)
        selected_distances = pairwise_distances(structure_descriptors_scaled[selected_indices])

        selection_info = {
            'method': selection_method,
            'aggregation': aggregation_method,
            'n_original': len(data),
            'n_selected': len(pruned_data),
            'pruning_ratio': pruning_ratio,
            'selected_indices': selected_indices.tolist(),
            'mean_distance_original': np.mean(all_distances[np.triu_indices_from(all_distances, k=1)]),
            'mean_distance_selected': np.mean(selected_distances[np.triu_indices_from(selected_distances, k=1)]),
            'descriptor_dim': structure_descriptors_scaled.shape[1],
            'random_seed': random_seed
        }
    else:
        selection_info = {
            'method': selection_method,
            'aggregation': aggregation_method,
            'n_original': len(data),
            'n_selected': len(pruned_data),
            'pruning_ratio': pruning_ratio,
            'selected_indices': selected_indices.tolist(),
            'descriptor_dim': structure_descriptors_scaled.shape[1],
            'random_seed': random_seed
        }

    return pruned_data, selection_info

def main():
    """Main function to demonstrate structure-level pruning with replicates"""

    # Load data
    print("Loading data...")
    data = loadfn("../data.json")
    print(f"Loaded {len(data)} structures")

    # Define pruning parameters
    pruning_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    aggregation_methods = ['mean_std', 'bag_of_atoms']
    selection_method = 'fps'
    n_replicates = 10

    # Create output directory
    output_dir = "replicates_structure_pruning"
    os.makedirs(output_dir, exist_ok=True)

    # Generate random seeds for replicates
    random.seed(42)  # For reproducible replicate generation
    replicate_seeds = [random.randint(0, 2**32-1) for _ in range(n_replicates)]

    # Process each combination with replicates
    for aggregation in aggregation_methods:
        for ratio in pruning_ratios:
            for replicate_idx in range(n_replicates):
                print(f"\n{'='*60}")
                print(f"Processing: {aggregation} aggregation, {ratio:.0%} retention, replicate {replicate_idx+1}/{n_replicates}")
                print(f"{'='*60}")

                # Perform pruning with random seed
                pruned_data, selection_info = structure_level_pruning(
                    data,
                    pruning_ratio=ratio,
                    aggregation_method=aggregation,
                    selection_method=selection_method,
                    random_seed=replicate_seeds[replicate_idx]
                )

                # Save results
                percentage = ratio * 100
                data_filename = f"{output_dir}/si_structures_{aggregation}_{percentage:.0f}percent_replicate{replicate_idx+1:02d}.json"
                info_filename = f"{output_dir}/si_structures_{aggregation}_{percentage:.0f}percent_replicate{replicate_idx+1:02d}_info.json"

                dumpfn(pruned_data, data_filename, indent=2)
                dumpfn(selection_info, info_filename, indent=2)

                print(f"Saved {len(pruned_data)} structures to {data_filename}")
                print(f"Selection info saved to {info_filename}")

                # Print summary statistics
                if 'mean_distance_original' in selection_info:
                    diversity_ratio = selection_info['mean_distance_selected'] / selection_info['mean_distance_original']
                    print(f"Diversity preservation: {diversity_ratio:.3f}")
                print(f"Random seed used: {replicate_seeds[replicate_idx]}")

if __name__ == "__main__":
    main()