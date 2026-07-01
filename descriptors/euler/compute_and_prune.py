"""
FPS-based structure pruning using the Euler Characteristic Curve (ECC) descriptor.

Unlike SOAP / Behler / Bispectrum (which produce *per-atom* descriptors that are
then aggregated to structure level via mean+std), the Euler characteristic is an
intrinsically *structure-level* topological quantity. For each frame we build a
periodic (minimum-image) Vietoris-Rips filtration over the atomic point cloud and
record the Euler characteristic

        chi(r) = V - E(r) + T(r)

as a function of the filtration radius r, where
    V    = number of atoms                     (0-simplices, constant per frame)
    E(r) = number of atom pairs with d <= r    (1-simplices)
    T(r) = number of triangles with all
           three edges <= r                    (2-simplices, VR flag complex)

Sampling chi(r) on a fixed radius grid yields one comparable feature vector (the
"Euler characteristic curve") per frame, which is standardized and fed to FPS.

The curve is normalized per atom by default so that cell size does not dominate
the descriptor (mixed 12- to 96-atom cells appear in data.json).

Prerequisites:
    data.json must be present at the project root (../../data.json).

Run from this directory:
    python compute_and_prune.py

Output: replicates_structure_pruning_modified/ (created in this directory)
"""

from __future__ import annotations
from monty.serialization import loadfn, dumpfn
from ase import Atoms
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from skmatter.sample_selection import FPS
import os
import random

# ── Euler Characteristic Curve parameters ──────────────────────────────────────
# Filtration radii (Angstrom) at which chi(r) is sampled. R_MAX ~ a few Si shells
# (nn ~2.35 A, 2nd nn ~3.84 A); comparable to the SOAP/Behler cutoffs.
R_MAX = 6.0
N_BINS = 64
NORMALIZE_PER_ATOM = True  # intensive descriptor -> comparable across cell sizes

FILTRATION_RADII = np.linspace(0.0, R_MAX, N_BINS)


def pymatgen_to_ase(structure):
    symbols = [site.species_string for site in structure]
    positions = structure.cart_coords
    cell = structure.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def euler_characteristic_curve(atoms, radii=FILTRATION_RADII, normalize=NORMALIZE_PER_ATOM):
    """Euler characteristic curve of the periodic Vietoris-Rips filtration.

    Returns a vector [chi(r_1), ..., chi(r_m)] with one entry per filtration radius.
    Distances use the minimum-image convention so the curve respects periodicity.
    """
    n_atoms = len(atoms)

    # Minimum-image pairwise distances (handles general / non-orthogonal cells).
    dist = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dist, np.inf)  # exclude self-pairs from every filtration

    chi = np.empty(len(radii))
    for k, r in enumerate(radii):
        adj = (dist <= r).astype(np.float64)  # 0/1 adjacency at this radius
        n_edges = adj.sum() / 2.0
        # Triangles in the flag complex: trace(A^3) counts each 3-clique 6 times.
        n_triangles = np.trace(adj @ adj @ adj) / 6.0
        chi[k] = n_atoms - n_edges + n_triangles

    if normalize:
        chi = chi / n_atoms

    return chi


def compute_descriptors(data):
    print(f"Computing Euler characteristic curves "
          f"(R_MAX={R_MAX} A, N_BINS={N_BINS}, normalize={NORMALIZE_PER_ATOM})...")

    structure_descriptors = []
    for i, d in enumerate(data):
        print(f"Processing structure {i+1}/{len(data)}", end='\r')
        atoms = pymatgen_to_ase(d["structure"])
        structure_descriptors.append(euler_characteristic_curve(atoms))

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
    descriptor_tag = 'euler_ecc'
    n_replicates = 10

    output_dir = "replicates_structure_pruning_modified"
    os.makedirs(output_dir, exist_ok=True)

    random.seed(42)
    replicate_seeds = [random.randint(0, 2**32 - 1) for _ in range(n_replicates)]

    # The Euler characteristic curves are deterministic, so compute them once and
    # reuse across replicates (only the FPS random initialization varies).
    structure_descriptors_scaled, _ = compute_descriptors(data)

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
            selection_info['descriptor'] = descriptor_tag

            percentage = ratio * 100
            data_fn = f"{output_dir}/si_structures_{descriptor_tag}_{percentage:.0f}percent_replicate{replicate_idx+1:02d}.json"
            info_fn = f"{output_dir}/si_structures_{descriptor_tag}_{percentage:.0f}percent_replicate{replicate_idx+1:02d}_info.json"

            dumpfn(pruned_data, data_fn, indent=2)
            dumpfn(selection_info, info_fn, indent=2)

            diversity_ratio = selection_info['mean_distance_selected'] / selection_info['mean_distance_original']
            print(f"Saved {len(pruned_data)} structures  diversity={diversity_ratio:.3f}")


if __name__ == "__main__":
    main()
