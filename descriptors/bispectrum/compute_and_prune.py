from __future__ import annotations
from monty.serialization import loadfn, dumpfn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from skmatter.sample_selection import FPS
import os
import random
from maml.describers import BispectrumCoefficients


# LAMMPS is required by BispectrumCoefficients (via maml). Update the path below
# to match your LAMMPS installation, or ensure `lmp` is already on your PATH.
os.environ['PATH'] = '/opt/homebrew/Cellar/lammps/20250722-update1/bin:' + os.environ.get('PATH', '')

ELEMENT_PROFILE = {"Si": {"r": 0.5, "w": 1}}


def _patch_maml_numpy2():
    """Make maml's LAMMPS dump reader work under numpy>=2.

    maml.apps.pes._lammps._read_dump defaults to the dtype string 'float_', an
    alias numpy removed in 2.0, so parsing dump.sna raises
    "data type 'float_' not understood". All internal callers use the module
    global, so replacing it here fixes every Bispectrum read. Idempotent.
    """
    import io
    from maml.apps.pes import _lammps
    if getattr(_lammps, "_np2_float_patched", False):
        return

    def _read_dump(file_name, dtype="float64"):
        if dtype == "float_":  # numpy<2 alias -> numpy>=2 canonical name
            dtype = "float64"
        with open(file_name) as f:
            lines = f.readlines()[9:]
        return np.loadtxt(io.StringIO("".join(lines)), dtype=dtype)

    _lammps._read_dump = _read_dump
    _lammps._np2_float_patched = True


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
    # Cache the RAW (pre-standardization) descriptor matrix in this folder so the
    # expensive LAMMPS Bispectrum evaluation runs only once; delete the .npy to
    # force a rebuild.
    cache = f"structure_descriptors_{aggregation_method}.npy"
    if os.path.exists(cache):
        print(f"Loading cached Bispectrum descriptors from {cache}")
        structure_descriptors = np.load(cache)
    else:
        print(f"Computing Bispectrum descriptors with {aggregation_method} aggregation...")
        _patch_maml_numpy2()
        structures = [d["structure"] for d in data]
        describer = BispectrumCoefficients(
            rcutfac=4.9,
            twojmax=8,
            element_profile=ELEMENT_PROFILE,
            quadratic=False,
            pot_fit=True,
            include_stress=False,
        )
        structure_descriptors = []
        for i, structure in enumerate(structures):
            print(f"Processing structure {i+1}/{len(structures)}", end='\r')
            atomic_descriptors = describer.transform_one(structure).values
            structure_descriptor = aggregate_atomic_descriptors(atomic_descriptors, method=aggregation_method)
            structure_descriptors.append(structure_descriptor)
        print()
        structure_descriptors = np.array(structure_descriptors)
        np.save(cache, structure_descriptors)
        print(f"Cached raw descriptors -> {cache}  shape={structure_descriptors.shape}")

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

    # Descriptors are deterministic, so compute them once and reuse across all
    # FPS replicates (only the FPS random_state changes per replicate).
    structure_descriptors_scaled, _ = compute_descriptors(data, aggregation_method)

    for replicate_idx in range(n_replicates):
        print(f"\n{'='*60}")
        print(f"Processing replicate {replicate_idx+1}/{n_replicates}  seed={replicate_seeds[replicate_idx]}")
        print(f"{'='*60}")

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
            data_fn = f"{output_dir}/si_structures_bispectrum_{aggregation_method}_{percentage:.0f}percent_replicate{replicate_idx+1:02d}.json"
            info_fn = f"{output_dir}/si_structures_bispectrum_{aggregation_method}_{percentage:.0f}percent_replicate{replicate_idx+1:02d}_info.json"

            dumpfn(pruned_data, data_fn, indent=2)
            dumpfn(selection_info, info_fn, indent=2)

            diversity_ratio = selection_info['mean_distance_selected'] / selection_info['mean_distance_original']
            print(f"Saved {len(pruned_data)} structures  diversity={diversity_ratio:.3f}")


if __name__ == "__main__":
    main()
