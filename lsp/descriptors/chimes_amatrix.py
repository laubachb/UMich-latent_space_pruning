"""ChIMES cluster-basis descriptor from a ChIMES least-squares design matrix (A matrix).

For a ChIMES fit the energy of a frame is a linear function of the summed 2/3/4-body
Chebyshev cluster polynomials, E = sum_p c_p A_p(frame). The energy row of the A matrix
is therefore the frame's "linear design row": the sum over all atoms (clusters) of the
basis functions. We use energy_row / natoms — the per-atom average of every basis
function, one value per ChIMES parameter — as the structure descriptor. This is the
exact ChIMES analogue of the per-atom "mean" aggregation of SOAP / bispectrum / MTP
basis functions and is what the ChIMES model itself sees.

Why not the force rows: per parameter the 3N force rows sum to zero over atoms
(Newton's third law), so their across-atom mean is identically zero and would only
add standardised round-off noise (the bug of the first Si pipeline).

A-matrix layout (chimes_lsq with FITENER true, FITSTRS false): frame f contributes
3 natoms_f force rows followed by 3 identical energy rows. Row labels are given in
b-labeled.txt (element symbol = force row, "+1" = energy row) and dim.txt holds
(n_params, n_rows). Frames are in the order of train.xyzf, which for the mlearn
fits is the order of training.json, except that cells too small for the ChIMES
cutoffs were replicated into supercells (natoms differs by an integer factor).
Replication leaves energy_row / natoms unchanged, so the descriptor is that of the
original frame; the alignment is verified frame by frame by the atom-count ratio.

Inputs per dataset ("chimes_dir" = external/chimes/<El>/): A.txt, b-labeled.txt,
dim.txt, train.xyzf. Output: (n_frames, n_params) per-atom energy-row matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def xyzf_natoms(path: Path) -> np.ndarray:
    n = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            k = int(line.split()[0])
            n.append(k)
            f.readline()
            for _ in range(k):
                f.readline()
    return np.array(n, dtype=int)


def frame_blocks(chimes_dir: Path) -> tuple[np.ndarray, int]:
    """Atom count per A-matrix frame (from train.xyzf), validated against b-labeled.txt and dim.txt."""
    nats = xyzf_natoms(chimes_dir / "train.xyzf")
    labels = np.loadtxt(chimes_dir / "b-labeled.txt", usecols=0, dtype=str)
    n_params, n_rows = map(int, (chimes_dir / "dim.txt").read_text().split())
    if len(labels) != n_rows:
        raise ValueError(f"{chimes_dir}: b-labeled.txt has {len(labels)} rows, dim.txt says {n_rows}")
    expected = int((3 * nats + 3).sum())
    if expected != n_rows:
        raise ValueError(f"{chimes_dir}: train.xyzf implies {expected} rows, A matrix has {n_rows}")
    r = 0
    for fi, n in enumerate(nats):
        blk = labels[r:r + 3 * n + 3]
        if np.any(blk[:3 * n] == "+1") or np.any(blk[3 * n:] != "+1"):
            raise ValueError(f"{chimes_dir}: frame {fi} row labels do not match 3N force + 3 energy rows")
        r += 3 * n + 3
    return nats, n_params


def energy_rows(chimes_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(energy_rows (n_frames, n_params), natoms (n_frames,)) parsed from A.txt."""
    nats, n_params = frame_blocks(chimes_dir)
    rows = np.empty((len(nats), n_params))
    with open(chimes_dir / "A.txt") as f:
        for fi, n in enumerate(nats):
            print(f"  chimes A-matrix frame {fi + 1}/{len(nats)}", end="\r", flush=True)
            for _ in range(3 * n):          # skip force rows
                f.readline()
            e = [np.array(f.readline().split(), dtype=float) for _ in range(3)]
            if any(len(x) != n_params for x in e):
                raise ValueError(f"frame {fi}: energy row has wrong number of columns")
            if not (np.allclose(e[0], e[1]) and np.allclose(e[0], e[2])):
                raise ValueError(f"frame {fi}: the 3 trailing rows are not identical energy rows")
            if np.isnan(e[0]).any():
                raise ValueError(f"frame {fi}: NaN in energy row")
            rows[fi] = e[0]
        if f.readline().strip():
            raise ValueError("A.txt has more rows than dim.txt / train.xyzf describe")
    print()
    return rows, nats


def structure_descriptors(data, chimes_dir: str | Path) -> np.ndarray:
    chimes_dir = Path(chimes_dir)
    rows, nats = energy_rows(chimes_dir)
    if len(nats) != len(data):
        raise ValueError(f"{len(nats)} A-matrix frames vs {len(data)} dataset frames")
    n_data = np.array([len(d["structure"]) for d in data])
    ratio = nats / n_data
    if np.any(ratio < 1) or np.any(np.abs(ratio - np.round(ratio)) > 1e-9):
        bad = np.where((ratio < 1) | (np.abs(ratio - np.round(ratio)) > 1e-9))[0]
        raise ValueError(f"A-matrix frames do not align with the dataset (atom-count ratio not "
                         f"an integer) at frames {bad[:10]}")
    n_rep = int((ratio > 1).sum())
    if n_rep:
        print(f"  {n_rep} frames were replicated supercells in the ChIMES fit "
              f"(factors {sorted(set(np.round(ratio[ratio > 1]).astype(int)))}); per-atom rows are unaffected")
    return rows / nats[:, None]
