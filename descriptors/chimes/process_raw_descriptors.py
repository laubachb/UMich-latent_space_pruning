"""
Parse a raw ChIMES least-squares design matrix (A matrix) into compact arrays.

Run from this directory:
    python process_raw_descriptors.py [path/to/A.txt] [path/to/natoms.txt]

Defaults: A.txt and natoms.txt in this directory.

A-matrix layout (verified on the 214-frame Si A.txt, 1101 parameters):
  natoms.txt has ONE LINE PER A-MATRIX ROW (the atom count of that row's frame).
  Each frame block = 3*natoms force rows (atom-major: x,y,z per atom) followed by
  3 identical energy rows (the frame's summed cluster-energy basis, repeated).
  Per column, the force rows sum to zero over atoms (Newton's third law), so the
  across-atom MEAN of the force rows is identically zero and carries no information.

Output: chimes_raw.npz with
  force_rows  (sum 3*natoms, n_params)  force-derivative rows, all frames stacked
  energy_rows (n_frames, n_params)      one energy row per frame
  natoms      (n_frames,)
  offsets     (n_frames + 1,)           frame f's force rows = force_rows[offsets[f]:offsets[f+1]]
"""

import sys
import numpy as np


def parse(a_path, natoms_path):
    natoms_per_row = np.loadtxt(natoms_path, dtype=int)
    n_rows = len(natoms_per_row)

    # Frame boundaries from the per-row atom counts
    nats = []
    r = 0
    while r < n_rows:
        n = natoms_per_row[r]
        block = 3 * n + 3
        if r + block > n_rows or np.any(natoms_per_row[r:r + block] != n):
            raise ValueError(f"Inconsistent frame block starting at row {r}")
        nats.append(n)
        r += block
    nats = np.array(nats)
    print(f"{len(nats)} frames, {n_rows} rows")

    n_force = int(3 * nats.sum())
    with open(a_path) as f:
        n_params = len(f.readline().split())
    print(f"{n_params} ChIMES parameters (columns)")

    force_rows = np.empty((n_force, n_params))
    energy_rows = np.empty((len(nats), n_params))
    offsets = np.concatenate([[0], np.cumsum(3 * nats)])

    with open(a_path) as f:
        for fi, n in enumerate(nats):
            block = np.array([np.array(next(f).split(), dtype=float) for _ in range(3 * n + 3)])
            if block.shape[1] != n_params:
                raise ValueError(f"Frame {fi}: {block.shape[1]} columns, expected {n_params}")
            forces, energy = block[:3 * n], block[3 * n:]

            if not (np.allclose(energy[0], energy[1]) and np.allclose(energy[0], energy[2])):
                raise ValueError(f"Frame {fi}: trailing 3 rows are not identical energy rows")
            scale = np.abs(forces).max()
            for c in range(3):
                resid = np.abs(forces[c::3].sum(0)).max()
                # absolute floor: perfect-crystal frames have ~zero force rows
                if resid > max(1e-8 * scale, 1e-9):
                    raise ValueError(f"Frame {fi}: force rows do not sum to zero (resid {resid:.2e})")
            if np.isnan(block).any():
                raise ValueError(f"Frame {fi}: NaN values")

            force_rows[offsets[fi]:offsets[fi + 1]] = forces
            energy_rows[fi] = energy[0]
        if f.readline().strip():
            raise ValueError("A matrix has more rows than natoms.txt describes")

    return force_rows, energy_rows, nats, offsets


if __name__ == "__main__":
    a_path = sys.argv[1] if len(sys.argv) > 1 else "A.txt"
    natoms_path = sys.argv[2] if len(sys.argv) > 2 else "natoms.txt"
    force_rows, energy_rows, nats, offsets = parse(a_path, natoms_path)
    np.savez("chimes_raw.npz", force_rows=force_rows, energy_rows=energy_rows,
             natoms=nats, offsets=offsets)
    print(f"Saved chimes_raw.npz  force_rows={force_rows.shape}  energy_rows={energy_rows.shape}")
