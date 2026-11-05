"""Atomic -> structure-level aggregation.

Per-atom descriptors (SOAP, Behler-Parrinello, bispectrum, MTP basis functions) are
aggregated to one vector per frame by concatenating the across-atom mean and
standard deviation of every feature ("mean_std"), the convention of the Si study.
"mean" is the per-atom average alone, i.e. the frame's linear design row divided
by the atom count — the aggregation used for the ChIMES A-matrix descriptor.
"""

from __future__ import annotations

import numpy as np


def aggregate(atomic: np.ndarray, method: str = "mean_std") -> np.ndarray:
    a = np.asarray(atomic, dtype=float)
    if method == "mean":
        return a.mean(axis=0)
    if method == "mean_std":
        return np.concatenate([a.mean(axis=0), a.std(axis=0)])
    if method == "sum":
        return a.sum(axis=0)
    raise ValueError(f"unknown aggregation '{method}'")


def stack_frames(per_frame_atomic: list[np.ndarray], method: str = "mean_std") -> np.ndarray:
    return np.array([aggregate(a, method) for a in per_frame_atomic])


def parallel_frames(fn, items, label: str, nproc: int | None = None) -> list:
    """Evaluate fn(item) for every item, in parallel when LSP_NPROC > 1 (default: all
    cores on a compute node, 4 elsewhere); order is preserved."""
    import os
    from multiprocessing import Pool
    n = nproc or int(os.environ.get("LSP_NPROC", 0)) or (os.cpu_count() if os.environ.get("SLURM_JOB_ID") else 4)
    n = max(1, min(n, len(items)))
    if n == 1:
        out = []
        for i, it in enumerate(items):
            print(f"  {label} {i + 1}/{len(items)}", end="\r", flush=True)
            out.append(fn(it))
        print()
        return out
    print(f"  {label}: {len(items)} frames on {n} processes", flush=True)
    with Pool(n) as pool:
        return pool.map(fn, items, chunksize=1)
