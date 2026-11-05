#!/usr/bin/env python3
"""
Build unsupervised/data/data_phase_categories.json from data.json.

Adds parallel labels without changing geometries or DFT outputs:
  - category       : legacy categorize_structure(...)
  - phase_category : categorize_phase(...)  (Diamond / Liquid / Vacancy-* / Surface)

Run from repo root:
    python analysis/umap/unsupervised/scripts/relabel_phase_categories.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
UNSUP_DIR = SCRIPT_DIR.parent
UMAP_DIR = UNSUP_DIR.parent
ROOT_DIR = UMAP_DIR.parents[1]
sys.path.insert(0, str(ROOT_DIR / "analysis"))

from categorization import categorize_phase, categorize_structure  # noqa: E402

SRC = ROOT_DIR / "data.json"
DST = UNSUP_DIR / "data" / "data_phase_categories.json"


def main() -> None:
    if not SRC.is_file():
        sys.exit(f"Missing {SRC}")

    DST.parent.mkdir(parents=True, exist_ok=True)

    with SRC.open() as f:
        data = json.load(f)

    out = []
    for entry in data:
        desc = entry.get("description") or ""
        new_entry = dict(entry)
        new_entry["category"] = categorize_structure(desc)
        new_entry["phase_category"] = categorize_phase(desc, entry.get("group"))
        out.append(new_entry)

    with DST.open("w") as f:
        json.dump(out, f, indent=2)

    counts = Counter(e["phase_category"] for e in out)
    print(f"Wrote {DST} ({len(out)} structures)")
    for cat, n in counts.most_common():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
