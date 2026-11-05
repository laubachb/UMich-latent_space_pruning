#!/usr/bin/env python3
"""Materialise the pruned training sets of a dataset as per-cut JSON files for model fitting.

    python scripts/export_cuts.py Si soap chimes --out /path/to/cuts

Writes <out>/<dataset>_<method>_<p>percent_replicate<NN>.json (a monty-serialised
list of the selected frames, same format as the mlearn training.json) and a
matching *_info.json with the selected frame indices and seed. FPS cuts are the
prefixes of the stored rankings (results/<dataset>/selection/<method>.json); the
random baseline stores its draws explicitly. Nothing here is tracked by git —
every cut is reproducible from the selection file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lsp import config, selection  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=config.available_datasets())
    ap.add_argument("methods", nargs="*", help="selection methods (default: all available)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    for m in a.methods or selection.available(a.dataset):
        files = selection.export_cuts(a.dataset, m, a.out / m)
        print(f"{a.dataset} {m}: {len(files)} cuts -> {a.out / m}")


if __name__ == "__main__":
    main()
