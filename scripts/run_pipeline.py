#!/usr/bin/env python3
"""Run the whole study for one dataset: descriptors -> FPS/random selection -> analyses -> plot data.

    python scripts/run_pipeline.py Si                 # everything, cached steps are skipped
    python scripts/run_pipeline.py Si --stage descriptors --descriptor soap mtp
    python scripts/run_pipeline.py Si --stage selection analysis export
    python scripts/run_pipeline.py Si --force         # recompute even if cached

Stages
    descriptors   every latent space in configs/<ds>.json plus the per-atom-mean variants
    selection     every sampler (fps, kmedoids, leverage; 25 replicates) in every latent
                  space + the random baseline
    analysis      composition, QUESTS information, distances, embeddings, dimensionality,
                  cross-space geometry + cut overlap, label-space coverage
    export        plotting/data/<ds>/ (FPS, mean+std) plus <ds>_mean/, <ds>_kmedoids/,
                  <ds>_leverage/, ... for the other aggregation / sampler variants
Then: cd plotting && ./make_figures.sh <ds>
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lsp import config, descriptors, selection  # noqa: E402
from lsp.analysis import composition, crossspace, dimensionality, distance, embeddings, export, labels, quests  # noqa: E402

STAGES = ["descriptors", "selection", "analysis", "export"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=config.available_datasets())
    ap.add_argument("--stage", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--descriptor", nargs="+", help="restrict to these latent spaces")
    ap.add_argument("--sampler", nargs="+", choices=selection.SAMPLERS, default=selection.SAMPLERS)
    ap.add_argument("--force", action="store_true", help="recompute cached results")
    ap.add_argument("--skip-failures", action="store_true", help="continue past a descriptor that fails")
    a = ap.parse_args()
    cfg = config.load(a.dataset)
    names = a.descriptor or descriptors.all_names(cfg)
    t0 = time.time()

    if "descriptors" in a.stage:
        for n in names:
            try:
                descriptors.compute(a.dataset, n, force=a.force)
            except Exception:
                if not a.skip_failures:
                    raise
                traceback.print_exc()
                print(f"[{a.dataset}] !! descriptor '{n}' failed, continuing", flush=True)
    if "selection" in a.stage:
        selection.run(a.dataset, "random", force=a.force)
        for s in a.sampler:
            for n in [n for n in names if n in descriptors.available(a.dataset)]:
                try:
                    selection.run(a.dataset, s, n, force=a.force)
                except Exception:
                    if not a.skip_failures:
                        raise
                    traceback.print_exc()
                    print(f"[{a.dataset}] !! selection '{s}-{n}' failed, continuing", flush=True)
    if "analysis" in a.stage:
        methods = selection.available(a.dataset)
        avail = descriptors.available(a.dataset)
        composition.run(a.dataset, methods)
        dimensionality.run(a.dataset, avail)
        distance.run(a.dataset, avail)
        embeddings.run(a.dataset, avail)
        quests.run(a.dataset, methods)
        crossspace.run(a.dataset)
        labels.run(a.dataset, methods)
    if "export" in a.stage:
        for s in a.sampler:
            export.run(a.dataset, sampler=s)
            if any(n in descriptors.ATOMIC for n in cfg["descriptors"]):
                export.run(a.dataset, variant="_mean", sampler=s)
    print(f"[{a.dataset}] pipeline stages {a.stage} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
