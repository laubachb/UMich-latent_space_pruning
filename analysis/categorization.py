"""
Structure categorization schemes for the Si data.json dataset.

Two parallel label sets:
  - categorize_structure / CATEGORIES
      Legacy composition-analysis labels (Strained, High Temp, …).
  - categorize_phase / PHASE_CATEGORIES
      Phase-oriented labels (Strained/Unstrained Diamond, Liquid, Vacancy-*, Surface).
"""

from __future__ import annotations

import re

CATEGORIES = [
    "Strained",
    "High Temp (3374K)",
    "Normal (300K)",
    "Vacancy (300K)",
    "Surface",
    "Other",
]

PHASE_CATEGORIES = [
    "Vacancy-300K",
    "Vacancy-3374K",
    "Surface",
    "Strained Diamond",
    "Unstrained Diamond",
    "Liquid",
]

# Coarse labels for supervised UMAP (merge vacancy temps + diamond strain).
PHASE_CATEGORIES_COARSE = [
    "Diamond",
    "Vacancy",
    "Liquid",
    "Surface",
]

# AIMD "normal" frames at these temperatures are treated as diamond-like solid.
DIAMOND_MAX_TEMP_K = 1518


def categorize_structure(description: str) -> str:
    """Legacy rules (analysis/normalized_category_composition.py)."""
    desc = description.lower()
    if "strain" in desc or "mode" in desc:
        return "Strained"
    if "vacancy" in desc:
        return "High Temp (3374K)" if "3374 K" in description else "Vacancy (300K)"
    if "surface" in desc or "si_mp-149" in desc:
        return "Surface"
    if "3374 K" in description:
        return "High Temp (3374K)"
    if "300 K" in description:
        return "Normal (300K)"
    return "Other"


def _temperature_k(description: str) -> int | None:
    match = re.search(r"at (\d+) K", description)
    return int(match.group(1)) if match else None


def _is_strained_diamond(description: str) -> bool:
    """Elastic / strain-mode cells (legacy Strained bin)."""
    desc = description.lower()
    return "strain" in desc or "mode" in desc


def categorize_phase(description: str, group: str | None = None) -> str:
    """
    Phase-oriented labels (parallel to categorize_structure).

    Rules:
      - Vacancy AIMD at 300 K      → Vacancy-300K
      - Vacancy AIMD at 3374 K     → Vacancy-3374K
      - Surface slabs              → Surface
      - Strained diamond cells     → Strained Diamond
        (Si 2x2x2 strain / mode descriptions; vacancies/surfaces excluded)
      - Unstrained diamond, T ≤ 1518 K → Unstrained Diamond
        (ground state + normal AIMD ≤ 1518 K)
      - T > 1518 K, not Vacancy-3374K → Liquid
        (normal AIMD at 2530 K / 3374 K)
    """
    desc = description or ""
    desc_l = desc.lower()
    group = group or ""

    is_vacancy = "vacancy" in desc_l or group == "Vacancy"
    if is_vacancy:
        temp = _temperature_k(desc)
        if temp == 300:
            return "Vacancy-300K"
        if temp == 3374:
            return "Vacancy-3374K"
        raise ValueError(f"Unrecognized vacancy temperature in: {desc!r}")

    is_surface = (
        "surface" in desc_l
        or "si_mp-149" in desc_l
        or group == "Surface"
    )
    if is_surface:
        return "Surface"

    if _is_strained_diamond(desc):
        return "Strained Diamond"

    temp = _temperature_k(desc)
    # Ground state (no T) or normal AIMD at T ≤ 1518 K
    if temp is None or temp <= DIAMOND_MAX_TEMP_K:
        return "Unstrained Diamond"
    return "Liquid"


def collapse_phase_vacancy(phase_category: str) -> str:
    """Map Vacancy-300K / Vacancy-3374K → Vacancy; leave other phase labels as-is."""
    if phase_category.startswith("Vacancy"):
        return "Vacancy"
    return phase_category


def collapse_phase_coarse(phase_category: str) -> str:
    """
    Coarse supervision labels: {Diamond, Vacancy, Liquid, Surface}.

    Collapses vacancy temperatures and strained/unstrained diamond.
    """
    if phase_category.startswith("Vacancy"):
        return "Vacancy"
    if phase_category.endswith("Diamond") or phase_category == "Diamond":
        return "Diamond"
    return phase_category


def phase_targets(
    phase_categories: list[str],
    label_set: list[str] | None = None,
    collapse_vacancy: bool = False,
    collapse_coarse: bool = False,
) -> list[str]:
    """
    Return supervision label strings.

    collapse_coarse=True → {Diamond, Vacancy, Liquid, Surface}
    collapse_vacancy=True → vacancy temps only collapsed
    otherwise → full PHASE_CATEGORIES
    """
    if collapse_coarse:
        return [collapse_phase_coarse(c) for c in phase_categories]
    if collapse_vacancy:
        return [collapse_phase_vacancy(c) for c in phase_categories]
    if label_set is not None:
        allowed = set(label_set)
        for c in phase_categories:
            if c not in allowed:
                raise ValueError(f"Unexpected phase label {c!r}; expected one of {label_set}")
    return list(phase_categories)
