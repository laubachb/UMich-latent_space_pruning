"""Structure-level descriptor registry with on-disk caching.

compute(dataset, name) returns the RAW (unstandardised) (n_frames, n_features)
matrix of one latent space and caches it as results/<dataset>/descriptors/<name>.npy.
Atom-centred descriptors also cache their per-atom values
(results/<dataset>/descriptors/<name>_atomic.npz: `X` stacked over frames, `offsets`)
so other aggregations can be formed without re-evaluating anything.

Latent spaces
    soap                SOAP power spectrum, mean + std over atoms          (dscribe)
    behler              Behler-Parrinello G2 + G4, mean + std over atoms     (numpy)
    bispectrum          bispectrum coefficients, mean + std over atoms       (maml + LAMMPS)
    mtp                 MTP basis functions, mean + std over atoms           (numpy)
    chimes              ChIMES A-matrix energy row / natoms (per-atom mean)  (A.txt)
    chimes_fingerprint  full-frame ChIMES cluster-graph fingerprint, alpha=1 (LAMMPS + tool)
    <name>_mean         per-atom MEAN only of an atom-centred descriptor (= its linear
                        design row / natoms, the aggregation a linear or average-kernel
                        model actually sees; the analogue of `chimes`) — from the cache
Standardisation (z-score per feature) is applied downstream.
"""

from __future__ import annotations

import json
import time

import numpy as np

from .. import config, data, paths
from .aggregate import aggregate

ATOMIC = ("soap", "behler", "bispectrum", "mtp")

LABELS = {
    "random": "Random",
    "soap": "SOAP",
    "behler": "Behler–Parrinello",
    "bispectrum": "Bispectrum",
    "mtp": "MTP",
    "chimes": "ChIMES",
    "chimes_fingerprint": "ChIMES fingerprint",
}

COLORS = {
    "random": "#555555",
    "soap": "#3A86FF",
    "behler": "#06A77D",
    "bispectrum": "#F77F00",
    "mtp": "#8338EC",
    "chimes": "#E63946",
    "chimes_fingerprint": "#A4161A",
}


AGGREGATION_SUFFIX = {"": "mean_std", "_mean": "mean"}


def split_name(name: str) -> tuple[str, str]:
    """'soap' -> ('soap', 'mean_std'); 'soap_mean' -> ('soap', 'mean')."""
    for suf, agg in AGGREGATION_SUFFIX.items():
        if suf and name.endswith(suf) and name[:-len(suf)] in ATOMIC:
            return name[:-len(suf)], agg
    return name, "mean_std"


def label(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    base, agg = split_name(name)
    return LABELS.get(base, base) + (" (per-atom mean)" if agg == "mean" else "")


def _atomic_cache(dataset: str, name: str):
    return paths.results_dir(dataset, "descriptors") / f"{name}_atomic.npz"


def _matrix_cache(dataset: str, name: str):
    return paths.results_dir(dataset, "descriptors") / f"{name}.npy"


def load_atomic(dataset: str, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    f = _atomic_cache(dataset, name)
    if not f.exists():
        return None
    z = np.load(f)
    return z["X"], z["offsets"]


def _build_atomic(dataset: str, name: str, frames, cfg) -> list[np.ndarray]:
    params = config.descriptor_params(cfg, name)
    el = cfg["element"]
    if name == "soap":
        from . import soap as m
    elif name == "behler":
        from . import behler as m
    elif name == "bispectrum":
        from . import bispectrum as m
    elif name == "mtp":
        from . import mtp as m
    else:
        raise KeyError(name)
    per_frame = m.atomic_descriptors(frames, params, el)
    offsets = np.concatenate([[0], np.cumsum([len(a) for a in per_frame])])
    np.savez_compressed(_atomic_cache(dataset, name), X=np.vstack(per_frame), offsets=offsets)
    return per_frame


def compute(dataset: str, name: str, force: bool = False) -> np.ndarray:
    """Raw structure-level descriptor matrix (cached)."""
    cache = _matrix_cache(dataset, name)
    if cache.exists() and not force:
        return np.load(cache)
    cfg = config.load(dataset)
    frames = data.load_frames(dataset)
    t0 = time.time()
    print(f"[{dataset}] computing '{name}' descriptors")
    base, agg = split_name(name)
    if base in ATOMIC:
        cached = None if force else load_atomic(dataset, base)
        if cached is None:
            per_frame = _build_atomic(dataset, base, frames, cfg)
        else:
            A, off = cached
            per_frame = [A[off[i]:off[i + 1]] for i in range(len(off) - 1)]
        X = np.array([aggregate(a, agg) for a in per_frame])
    elif name == "chimes":
        npz = cfg.get("chimes", {}).get("npz")
        if npz:   # pre-parsed energy rows (legacy Si study: chimes_raw.npz of process_raw_descriptors.py)
            raw = np.load(paths.ROOT / npz)
            nat = raw["natoms"]
            if len(nat) != len(frames) or any(len(d["structure"]) != n for d, n in zip(frames, nat)):
                raise ValueError("legacy ChIMES npz does not align with the dataset")
            X = raw["energy_rows"] / nat[:, None]
        else:
            from . import chimes_amatrix
            X = chimes_amatrix.structure_descriptors(frames, cfg["chimes_dir"])
    elif name == "chimes_fingerprint":
        from . import chimes_fingerprint
        X = chimes_fingerprint.structure_descriptors(frames, config.descriptor_params(cfg, name),
                                                     cfg["element"], cfg["chimes_dir"])
    else:
        raise KeyError(f"unknown descriptor '{name}'")
    if X.shape[0] != len(frames):
        raise RuntimeError(f"{name}: {X.shape[0]} rows for {len(frames)} frames")
    np.save(cache, X)
    meta = dict(dataset=dataset, descriptor=name, shape=list(X.shape), seconds=round(time.time() - t0, 1),
                params=cfg.get(name, {}), n_nan_rows=int((~np.isfinite(X).all(axis=1)).sum()))
    with open(cache.with_suffix(".json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[{dataset}] '{name}': {X.shape} in {meta['seconds']} s -> {cache}")
    return X


def all_names(cfg: dict) -> list[str]:
    """Every latent space of a dataset: configured descriptors plus the per-atom-mean
    variants of the atom-centred ones (main order first, then the variants)."""
    names = list(cfg["descriptors"])
    return names + [f"{n}_mean" for n in names if n in ATOMIC]


def available(dataset: str) -> list[str]:
    d = paths.results_dir(dataset, "descriptors")
    return [n for n in all_names(config.load(dataset)) if (d / f"{n}.npy").exists()]


def standardize(X: np.ndarray) -> np.ndarray:
    """z-score per feature; zero-variance features collapse to 0; NaN rows stay NaN."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd
