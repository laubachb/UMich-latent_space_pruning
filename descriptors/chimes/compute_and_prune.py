"""
FPS-based structure pruning using ChIMES descriptors.

Prerequisites:
    1. Generate frames_descriptors.pkl by running process_raw_descriptors.py first.
    2. data.json must be present at the project root (../../data.json).

Run from this directory:
    python compute_and_prune.py

Output: replicates_structure_pruning_modified/ (created in this directory)
"""

from __future__ import annotations
from monty.serialization import loadfn, dumpfn
from skmatter.sample_selection import FPS
from sklearn.metrics import pairwise_distances
import numpy as np
import random
import os
import pickle


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

    print("Loading ChIMES frame descriptors...")
    with open("frames_descriptors.pkl", "rb") as f:
        frame_descriptors = pickle.load(f)
    print(f"Loaded {len(frame_descriptors)} frame matrices")

    # Match data.json entries to descriptor matrices via frame_index field
    mean_std_descriptors = []
    matched_indices = []
    frames_with_zero_std = []

    for i, d in enumerate(data):
        frame_index = d.get("frame_index", i)
        if frame_index not in frame_descriptors:
            print(f"Warning: frame index {frame_index} not found; skipping entry {i}")
            continue

        matrix = np.array(frame_descriptors[frame_index])
        mean_desc = np.mean(matrix, axis=0)
        std_desc = np.std(matrix, axis=0)

        if np.any(std_desc == 0):
            zero_count = int(np.sum(std_desc == 0))
            frames_with_zero_std.append((frame_index, zero_count))

        mean_std_descriptors.append(np.concatenate([mean_desc, std_desc]))
        matched_indices.append(i)

    mean_std_descriptors = np.array(mean_std_descriptors)
    print(f"Matched {len(mean_std_descriptors)} descriptors out of {len(data)}")

    if frames_with_zero_std:
        print(f"Warning: {len(frames_with_zero_std)} frames have zero-std descriptor components.")

    # Standardize; guard against zero-std columns
    mean = np.mean(mean_std_descriptors, axis=0)
    std = np.std(mean_std_descriptors, axis=0)
    std[std == 0] = 1.0
    structure_descriptors_scaled = (mean_std_descriptors - mean) / std
    structure_descriptors_scaled = np.nan_to_num(structure_descriptors_scaled, nan=0.0)

    data_matched = [data[i] for i in matched_indices]

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

        fps = FPS(
            initialize='random',
            n_to_select=len(structure_descriptors_scaled),
            random_state=replicate_seeds[replicate_idx],
        )
        fps.fit(structure_descriptors_scaled)
        fps_ranking = fps.selected_idx_
        print(f"FPS ranking computed: {len(fps_ranking)} structures")

        for ratio in pruning_ratios:
            print(f"\nPruning to {ratio:.0%} retention...")

            pruned_data, selection_info = structure_level_pruning(
                data_matched, structure_descriptors_scaled, fps_ranking, pruning_ratio=ratio
            )
            selection_info['random_seed'] = replicate_seeds[replicate_idx]
            selection_info['aggregation'] = aggregation_method

            percentage = ratio * 100
            data_fn = f"{output_dir}/si_structures_chimes_{aggregation_method}_{percentage:.0f}percent_replicate{replicate_idx+1:02d}.json"
            info_fn = f"{output_dir}/si_structures_chimes_{aggregation_method}_{percentage:.0f}percent_replicate{replicate_idx+1:02d}_info.json"

            dumpfn(pruned_data, data_fn, indent=2)
            dumpfn(selection_info, info_fn, indent=2)

            diversity_ratio = selection_info['mean_distance_selected'] / selection_info['mean_distance_original']
            print(f"Saved {len(pruned_data)} structures  diversity={diversity_ratio:.3f}")


if __name__ == "__main__":
    main()
