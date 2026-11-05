#!/usr/bin/env python3
"""
Write the two label sets of every Si frame to unsupervised/data/data_phase_categories.json.

Labels only (no geometries or DFT outputs, which live in the mlearn training.json):
  - index           : frame index in external/mlearn/Si/training.json
  - description     : the mlearn description
  - group           : the mlearn group (AIMD-NVT / Vacancy / Elastic / Surface)
  - category        : legacy categorize_structure(...)
  - phase_category  : categorize_phase(...)  (Diamond / Liquid / Vacancy-* / Surface)

The figure generator applies the same rules directly, so this file is a readable record
of the labelling, not an input.

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
sys.path.insert(0, str(SCRIPT_DIR))

import umap_helpers as h  # noqa: E402
from categorization import categorize_phase, categorize_structure  # noqa: E402

DST = UNSUP_DIR / "data" / "data_phase_categories.json"


def main() -> None:
    frames = h.load_frames()
    out = []
    for i, entry in enumerate(frames):
        desc = entry.get("description") or ""
        out.append(dict(index=i, description=desc, group=entry.get("group"),
                        category=categorize_structure(desc),
                        phase_category=categorize_phase(desc, entry.get("group"))))
    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w") as f:
        json.dump(out, f, indent=1)
    counts = Counter(e["phase_category"] for e in out)
    print(f"Wrote {DST} ({len(out)} frames)")
    for cat, n in counts.most_common():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
