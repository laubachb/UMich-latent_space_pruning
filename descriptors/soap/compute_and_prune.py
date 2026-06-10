from __future__ import annotations
from monty.serialization import loadfn, dumpfn
from dscribe.descriptors import SOAP
from ase import Atoms
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from skmatter.sample_selection import FPS
import os
import random

# SOAP parameters for Si — adjust r_cut, n_max, l_max as needed.
SOAP_PARAMS = dict(
    species=["Si"],
    r_cut=5.0,
    n_max=9,
    l_max=9,
    periodic=True,
)


def pymatgen_to_ase(structure):
    symbols = [site.species_string for site in structure]
    positions = structure.cart_coords
    cell = structure.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def aggregate_atomic_descriptors(atomic_descriptors, method='mean_std'):
    if method == 'mean':
        return np.mean(atomic_descriptors, axis=0)
    elif method == 'mean_std':
        mean_desc = np.mean(atomic_descriptors, axis=0)
        std_desc = np.std(atomic_descriptors, axis=0)
        return np.concatenate([mean_desc, std_desc])
    elif method == 'sum':
        return np.sum(atomic_descriptors, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_descriptors(data, aggregation_method='mean_std'):
    print(f"Computing SOAP descriptors with {aggregation_method} aggregation...")

    soap = SOAP(**SOAP_PARAMS)

    structure_descriptors = []
    for i, d in enumerate(data):
        print(f"Processing structure {i+1}/{len(data)}", end='\r')
        atoms = pymatgen_to_ase(d["structure"])
        atomic_descriptors = soap.create(atoms)  # shape: (n_atoms, n_features)
        structure_descriptor = aggregate_atomic_descriptors(atomic_descriptors, method=aggregation_method)
        structure_descriptors.append(structure_descriptor)

    print()
    structure_descriptors = np.array(structure_descriptors)

    scaler = StandardScaler()
    structure_descriptors_scaled = scaler.fit_transform(structure_descriptors)

    print(f"Structure descriptors shape: {structure_descriptors_scaled.shape}")
    return structure_descriptors_scaled, scaler


def structure_level_pruning(data, structure_descriptors_scaled, fps_ranking, pruning_ratio):
    print(f"Pruning to {pruning_ratio:.1%} of {len(data)} structures using pre-computed FPS ranking...")

    n_select = max(1, int(len(data) * pruning_ratio))
    selected_indices = fps_ranking[:n_select]
    pruned_data = [data[i] for i in selected_indices]

    all_distances = pairwise_distances(structure_descriptors_scaled)
    selected_distances = pairwise_distances(structure_descriptors_scaled[selected_indices])

    selection_info = {
        'method': 'fps',
        'n_original': len(data),
        'n_selected': len(pruned_data),
        'pruning_ratio': pruning_ratio,
        'selected_indices': selected_indices.tolist(),
        'mean_distance_original': float(np.mean(all_distances[np.triu_indices_from(all_distances, k=1)])),
        'mean_distance_selected': float(np.mean(selected_distances[np.triu_indices_from(selected_distances, k=1)])),
        'descriptor_dim': structure_descriptors_scaled.shape[1],
    }

    return pruned_data, selection_info


def main():
    print("Loading data...")
    data = loadfn("../../data.json")
    print(f"Loaded {len(data)} structures")

    pruning_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    aggregation_method = 'mean_std'
    n_replicates = 10

    output_dir = "replicates_structure_pruning_modified"
    os.makedirs(output_dir, exist_ok=True)

    random.seed(42)
    replicate_seeds = [random.randint(0, 2**32 - 1) for _ in range(n_replicates)]

    for replicate_idx in range(n_replicates):
        print(f"\n{'='*60}")
        print(f"Processing replicate {replicate_idx+1}/{n_replicates}  seed={replicate_seeds[replicate_idx]}")
        print(f"{'='*60}")

        structure_descriptors_scaled, _ = compute_descriptors(data, aggregation_method)

        print(f"Performing FPS with seed {replicate_seeds[replicate_idx]}...")
        fps = FPS(initialize='random', n_to_select=len(data), random_state=replicate_seeds[replicate_idx])
        fps.fit(structure_descriptors_scaled)
        fps_ranking = fps.selected_idx_
        print(f"FPS ranking computed: {len(fps_ranking)} structures")

        for ratio in pruning_ratios:
            print(f"\nPruning to {ratio:.0%} retention...")

            pruned_data, selection_info = structure_level_pruning(
                data, structure_descriptors_scaled, fps_ranking, pruning_ratio=ratio
            )
            selection_info['random_seed'] = replicate_seeds[replicate_idx]
            selection_info['aggregation'] = aggregation_method

            percentage = ratio * 100
            data_fn = f"{output_dir}/si_structures_soap_{aggregation_method}_{percentage:.0f}percent_replicate{replicate_idx+1:02d}.json"
            info_fn = f"{output_dir}/si_structures_soap_{aggregation_method}_{percentage:.0f}percent_replicate{replicate_idx+1:02d}_info.json"

            dumpfn(pruned_data, data_fn, indent=2)
            dumpfn(selection_info, info_fn, indent=2)

            diversity_ratio = selection_info['mean_distance_selected'] / selection_info['mean_distance_original']
            print(f"Saved {len(pruned_data)} structures  diversity={diversity_ratio:.3f}")


if __name__ == "__main__":
    main()
