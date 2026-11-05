#!/usr/bin/env python3
"""Download the mlearn benchmark datasets (Zuo et al. 2020) into external/mlearn/<El>/training.json.

    python scripts/fetch_mlearn_data.py            # all six elements
    python scripts/fetch_mlearn_data.py Si Cu      # a subset

Source: https://github.com/materialsvirtuallab/mlearn (BSD-3 licence), data/<El>/training.json.
The files are not tracked by this repository.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/{el}/training.json"
ELEMENTS = ["Si", "Cu", "Ge", "Li", "Mo", "Ni"]


def main(elements):
    for el in elements:
        out = ROOT / "external" / "mlearn" / el / "training.json"
        if out.exists():
            print(f"{el}: {out} exists, skipping")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        url = BASE.format(el=el)
        print(f"{el}: {url} -> {out}")
        urllib.request.urlretrieve(url, out)


if __name__ == "__main__":
    main(sys.argv[1:] or ELEMENTS)
