"""Per-dataset configuration.

A dataset config (configs/<name>.json) names the element, where its frames and
ChIMES inputs live, and the hyperparameters of every descriptor evaluated on it.
The six mlearn configs use the per-element hyperparameters of the expert-optimised
models of Zuo et al., J. Phys. Chem. A 124, 731 (2020) (see README, "Descriptor
hyperparameters"); si_legacy reproduces the parameters of the original Si study.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import paths

# Descriptors in the order they appear in figures/tables.
DESCRIPTOR_ORDER = ["soap", "behler", "bispectrum", "mtp", "chimes", "chimes_fingerprint"]


def available_datasets() -> list[str]:
    return sorted(p.stem for p in paths.CONFIGS.glob("*.json")
                  if not p.stem.startswith("local_paths"))


def load(name: str) -> dict:
    f = paths.CONFIGS / f"{name}.json"
    if not f.exists():
        raise FileNotFoundError(f"no dataset config {f}; available: {available_datasets()}")
    with open(f) as fh:
        cfg = json.load(fh)
    cfg["name"] = name
    cfg.setdefault("descriptors", DESCRIPTOR_ORDER)
    # resolve relative paths against the repository root
    for key in ("frames", "chimes_dir"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(paths.ROOT / cfg[key])
    return cfg


def descriptor_params(cfg: dict, descriptor: str) -> dict:
    try:
        return cfg[descriptor]
    except KeyError:
        raise KeyError(f"dataset '{cfg['name']}' has no '{descriptor}' section") from None
