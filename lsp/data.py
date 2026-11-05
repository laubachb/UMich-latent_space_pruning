"""Dataset loading, structure categories and per-frame metadata.

Frames come from the mlearn benchmark JSON files (Zuo et al. 2020;
https://github.com/materialsvirtuallab/mlearn, data/<El>/training.json): a
monty-serialised list of dicts with a pymatgen `structure`, a free-text
`description`, a `group` (AIMD-NVT / Vacancy / Elastic / Surface) and DFT
`outputs` (energy, forces, virial_stress).

Categories are parsed from the description with one generic rule set so every
element gets the same scheme:

    Surface            slab from Si_surface_mp/... (any Miller index, reconstructed or not)
    Strained           elastic strain modes, the relaxed/ground-state crystal
    Vacancy <T> K      AIMD with a single vacancy at temperature T
    Bulk <T> K         defect-free AIMD at temperature T (Li's shrunken / expanded
                       volumes are folded into the same temperature bin; the volume
                       state is kept as separate metadata)

Category order (used for stacking and colours): Surface, Vacancy (cold -> hot),
Bulk (hot -> cold), Strained — the scheme of the Si study.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from monty.serialization import loadfn

from . import config

_TEMP_RE = re.compile(r"(\d+)\s*K\b")


@lru_cache(maxsize=8)
def load_frames(dataset: str) -> list:
    cfg = config.load(dataset)
    data = loadfn(cfg["frames"])
    if "n_frames" in cfg and len(data) != cfg["n_frames"]:
        raise ValueError(f"{dataset}: {len(data)} frames in {cfg['frames']}, "
                         f"config says {cfg['n_frames']}")
    return data


def to_ase(structure):
    from ase import Atoms
    return Atoms(symbols=[site.species_string for site in structure],
                 positions=structure.cart_coords, cell=structure.lattice.matrix, pbc=True)


def parse_category(description: str) -> tuple[str, float | None]:
    """(kind, temperature) with kind in {Surface, Strained, Vacancy, Bulk}."""
    d = description.lower()
    if "surface" in d:
        return "Surface", None
    if "strain" in d or "mode" in d or "ground state" in d or "relaxed" in d:
        return "Strained", None
    m = _TEMP_RE.search(description)
    if m is None:
        raise ValueError(f"cannot categorise frame description {description!r}")
    return ("Vacancy" if "vacancy" in d else "Bulk"), float(m.group(1))


def volume_state(description: str) -> str:
    d = description.lower()
    for s in ("shrunken", "expanded"):
        if s in d:
            return s
    return "normal"


def category_name(kind: str, temp: float | None) -> str:
    return kind if temp is None else f"{kind} {int(temp)} K"


def categories(data) -> tuple[list[str], np.ndarray]:
    """Ordered category names for this dataset and the category id of every frame."""
    parsed = [parse_category(d["description"]) for d in data]
    vac_t = sorted({t for k, t in parsed if k == "Vacancy"})
    bulk_t = sorted({t for k, t in parsed if k == "Bulk"}, reverse=True)
    names = (["Surface"] + [category_name("Vacancy", t) for t in vac_t]
             + [category_name("Bulk", t) for t in bulk_t] + ["Strained"])
    names = [n for n in names if any(category_name(k, t) == n for k, t in parsed)]
    ids = np.array([names.index(category_name(k, t)) for k, t in parsed])
    return names, ids


def metadata(data) -> dict:
    """Per-frame metadata arrays (categorical + continuous physical handles)."""
    names, ids = categories(data)
    temps, groups, vol, natoms, epa, stress = [], [], [], [], [], []
    for d in data:
        kind, t = parse_category(d["description"])
        temps.append(np.nan if t is None else t)
        groups.append(d.get("group", "Unknown"))
        vol.append(volume_state(d["description"]))
        n = len(d["structure"])
        natoms.append(n)
        out = d.get("outputs", {}) or {}
        e = out.get("energy")
        epa.append(np.nan if e is None else e / n)
        vs = out.get("virial_stress")
        stress.append(np.nan if vs is None or len(vs) < 3 else float(np.mean(np.asarray(vs, float)[:3])))
    return {
        "category": np.array([names[i] for i in ids]),
        "category_id": ids,
        "group": np.array(groups),
        "volume_state": np.array(vol),
        "temperature_K": np.array(temps, float),
        "num_atoms": np.array(natoms, int),
        "energy_per_atom": np.array(epa, float),
        "mean_normal_stress": np.array(stress, float),
        "description": np.array([d["description"] for d in data]),
    }
