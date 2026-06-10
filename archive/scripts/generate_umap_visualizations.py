from __future__ import annotations
from monty.serialization import loadfn
from dscribe.descriptors import SOAP
from ase import Atoms
import numpy as np
from sklearn.preprocessing import StandardScaler
from maml.describers import BPSymmetryFunctions, BispectrumCoefficients
import matplotlib.pyplot as plt
import umap
import os

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
        atomic_descriptors: 2D array of atomic descriptors for one structure
        method: aggregation method ('mean', 'mean_std', 'sum')

    Returns:
        structure_descriptor: 1D array representing the entire structure
    """
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

def compute_behler_descriptors(structures):
    """Compute Behler symmetry function descriptors"""
    print("Computing Behler descriptors...")

    # Add LAMMPS to PATH
    os.environ['PATH'] = '/opt/homebrew/Cellar/lammps/20250722-update1/bin:' + os.environ.get('PATH', '')

    # Initialize BPSymmetryFunctions with parameters from behler/replicate_test_modified.py
    describer = BPSymmetryFunctions(
        cutoff=5.5,
        r_etas=[0.01, 0.02, 0.05],
        r_shift=[4.0],
        a_etas=[0.01, 0.02, 0.05],
        zetas=[1.0, 16.0, 2.0, 4.0],
        lambdas=[1, -1]
    )

    structure_descriptors = []
    for i, structure in enumerate(structures):
        print(f"Processing structure {i+1}/{len(structures)}", end='\r')
        atomic_descriptors_df = describer.transform_one(structure)
        atomic_descriptors = atomic_descriptors_df.values
        structure_descriptor = aggregate_atomic_descriptors(atomic_descriptors, method='mean_std')
        structure_descriptors.append(structure_descriptor)

    print()
    return np.array(structure_descriptors)

def compute_soap_descriptors(structures):
    """Compute SOAP descriptors"""
    print("Computing SOAP descriptors...")

    # Convert to ASE
    ase_structures = [pymatgen_to_ase(struct) for struct in structures]

    # Initialize SOAP with parameters from pruning/test.py
    soap = SOAP(
        species=["Si"],
        periodic=True,
        r_cut=5.1,
        n_max=8,
        l_max=8,
        average="off",
    )

    structure_descriptors = []
    for i, ase_structure in enumerate(ase_structures):
        print(f"Processing structure {i+1}/{len(ase_structures)}", end='\r')
        soap_descriptors = soap.create(ase_structure)
        structure_descriptor = aggregate_atomic_descriptors(soap_descriptors, method='mean_std')
        structure_descriptors.append(structure_descriptor)

    print()
    return np.array(structure_descriptors)

def compute_bispectrum_descriptors(structures):
    """Compute Bispectrum descriptors"""
    print("Computing Bispectrum descriptors...")

    element_profile = {"Si": {"r": 0.5, "w": 1}}

    # Add LAMMPS to PATH
    os.environ['PATH'] = '/opt/homebrew/Cellar/lammps/20250722-update1/bin:' + os.environ.get('PATH', '')

    # Initialize BispectrumCoefficients with parameters from bispectrum/test.py
    describer = BispectrumCoefficients(
        rcutfac=4.9,
        twojmax=8,
        element_profile=element_profile,
        quadratic=False,
        pot_fit=True,
        include_stress=False
    )

    structure_descriptors = []
    for i, structure in enumerate(structures):
        print(f"Processing structure {i+1}/{len(structures)}", end='\r')
        atomic_descriptors_df = describer.transform_one(structure)
        atomic_descriptors = atomic_descriptors_df.values
        structure_descriptor = aggregate_atomic_descriptors(atomic_descriptors, method='mean_std')
        structure_descriptors.append(structure_descriptor)

    print()
    return np.array(structure_descriptors)

def compute_umap_embedding(descriptors):
    """Compute UMAP embedding for given descriptors"""
    # Standardize descriptors
    scaler = StandardScaler()
    descriptors_scaled = scaler.fit_transform(descriptors)

    # Apply UMAP
    reducer = umap.UMAP(n_components=2, min_dist=0.9, random_state=42)
    embedding = reducer.fit_transform(descriptors_scaled)

    return embedding

def main():
    # Load data
    print("Loading data...")
    data = loadfn("data.json")
    structures = [d["structure"] for d in data]
    energies = [d["outputs"]["energy"] for d in data]
    print(f"Loaded {len(structures)} structures\n")

    # Check for cached descriptors
    descriptors_cache = "descriptors_cache.npz"
    if os.path.exists(descriptors_cache):
        print("Loading cached descriptors...")
        cache = np.load(descriptors_cache)
        behler_descriptors = cache['behler']
        soap_descriptors = cache['soap']
        bispectrum_descriptors = cache['bispectrum']
        print("Cached descriptors loaded.\n")
    else:
        # Compute descriptors
        behler_descriptors = compute_behler_descriptors(structures)
        soap_descriptors = compute_soap_descriptors(structures)
        bispectrum_descriptors = compute_bispectrum_descriptors(structures)

        # Save descriptors to cache
        print("\nSaving descriptors to cache...")
        np.savez(descriptors_cache,
                 behler=behler_descriptors,
                 soap=soap_descriptors,
                 bispectrum=bispectrum_descriptors)
        print(f"Descriptors cached to {descriptors_cache}")

    # Compute UMAP embeddings
    print("\nComputing UMAP embeddings...")
    behler_embedding = compute_umap_embedding(behler_descriptors)
    soap_embedding = compute_umap_embedding(soap_descriptors)
    bispectrum_embedding = compute_umap_embedding(bispectrum_descriptors)

    # Create figure with three subplots
    print("Creating combined visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Plot Behler
    scatter1 = axes[0].scatter(behler_embedding[:, 0], behler_embedding[:, 1],
                               c=energies, s=5, alpha=0.6, cmap='viridis')
    axes[0].set_title('Behler Symmetry Functions', fontsize=14)
    axes[0].set_xlabel('UMAP 1', fontsize=12)
    axes[0].set_ylabel('UMAP 2', fontsize=12)
    fig.colorbar(scatter1, ax=axes[0], label='Energy')

    # Plot SOAP
    scatter2 = axes[1].scatter(soap_embedding[:, 0], soap_embedding[:, 1],
                               c=energies, s=5, alpha=0.6, cmap='viridis')
    axes[1].set_title('SOAP', fontsize=14)
    axes[1].set_xlabel('UMAP 1', fontsize=12)
    axes[1].set_ylabel('UMAP 2', fontsize=12)
    fig.colorbar(scatter2, ax=axes[1], label='Energy')

    # Plot Bispectrum
    scatter3 = axes[2].scatter(bispectrum_embedding[:, 0], bispectrum_embedding[:, 1],
                               c=energies, s=5, alpha=0.6, cmap='viridis')
    axes[2].set_title('Bispectrum', fontsize=14)
    axes[2].set_xlabel('UMAP 1', fontsize=12)
    axes[2].set_ylabel('UMAP 2', fontsize=12)
    fig.colorbar(scatter3, ax=axes[2], label='Energy')

    plt.tight_layout()
    plt.savefig('umap_all_descriptors.png', dpi=300)
    plt.close()

    print("Saved visualization to umap_all_descriptors.png")
    print("\nAll visualizations completed!")

if __name__ == "__main__":
    main()
