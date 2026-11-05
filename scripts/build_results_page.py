#!/usr/bin/env python3
"""Build the results page (element / sampler tabs over the per-atom-mean figures).

    python scripts/build_results_page.py --out /path/lsp_results.html

Fills scripts/results_page/template.html with the summary numbers of every dataset
that has rendered figures (scripts/artifact_data.py). The page references the figures
as <tag>/<figure>.png, i.e. relative to the repository's figures/ directory; publish
it with those files alongside (tags <El>_mean, <El>_mean_kmedoids, <El>_mean_leverage).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import artifact_data  # noqa: E402
from lsp import config  # noqa: E402

FIGS = ["fig_dataset_composition", "fig_dataset_information", "fig_distance_analysis", "fig_crossspace",
        "fig_label_coverage"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = {}
    for ds in config.available_datasets():
        if ds == "si_legacy":
            continue
        b = artifact_data.dataset_block(ds)
        if b and b["tags"]:
            data[ds] = b
    page = (ROOT / "scripts" / "results_page" / "template.html").read_text().replace("__DATA__", json.dumps(data))
    a.out.write_text(page)
    files = [f"{ds}_mean{suffix}/{f}.png" for ds in data for suffix in ("", "_kmedoids", "_leverage") for f in FIGS]
    print(f"{a.out} ({len(page) // 1024} KB) for {sorted(data)}")
    print("figure files to publish alongside (relative to figures/):")
    print(json.dumps(files))


if __name__ == "__main__":
    main()
