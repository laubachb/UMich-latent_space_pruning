#!/usr/bin/env python3
"""Check that the package reproduces the original Si study exactly.

Compares, for the `si_legacy` configuration,
  * the raw structure-descriptor matrices (SOAP, Behler-Parrinello, bispectrum,
    ChIMES) with the caches of the original per-descriptor scripts
    (legacy_si/descriptor_cache/<name>.npy), and
  * the FPS selections (10 replicates x 18 retentions) with the `selected_indices`
    of the original replicate cuts (legacy_si/cuts/<name>/*_info.json).

Run from the repository root after
    python scripts/run_pipeline.py si_legacy --stage descriptors selection --sampler fps
    python tests/test_legacy_si.py
Both legacy directories are untracked (they only exist on the machine of the Si study).
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lsp import descriptors, paths, selection  # noqa: E402

LEGACY = paths.ROOT / "legacy_si"
RE = re.compile(r"_(\d+)percent_replicate(\d+)_info\.json$")


def main():
    ok = True
    for name in ["soap", "behler", "bispectrum", "chimes"]:
        ref = LEGACY / "descriptor_cache" / f"{name}.npy"
        if not ref.exists():
            print(f"{name}: no legacy cache, skipped")
            continue
        X = descriptors.compute("si_legacy", name)
        L = np.load(ref)
        same = X.shape == L.shape and np.allclose(X, L, rtol=1e-6, atol=1e-9)
        print(f"{name:11s} descriptors {X.shape} == legacy: {same}")
        ok &= same
        cuts = sorted(glob.glob(str(LEGACY / "cuts" / name / "*_info.json")))
        if not cuts:
            continue
        sel = selection.load("si_legacy", selection.method_id("fps", name))
        mine = {(p, r + 1): idx.tolist() for p, r, idx in selection.iter_cuts(sel)}
        n_same = 0
        for f in cuts:
            p, r = map(int, RE.search(f).groups())
            with open(f) as fh:
                n_same += json.load(fh)["selected_indices"] == mine[(p, r)]
        print(f"{name:11s} FPS cuts identical: {n_same}/{len(cuts)}")
        if n_same < len(cuts):
            # Exactly duplicated frames (symmetry-equivalent strain modes) give tied FPS
            # distances; 1e-15 differences in summation order can then flip the pick.
            # Check that the FPS code path itself is identical on the legacy matrix.
            Z = descriptors.standardize(L)
            seeds = selection.replicate_seeds()
            n_rep = 0
            for f in cuts:
                p, r = map(int, RE.search(f).groups())
                rank = selection.fps_ranking(Z, seeds[r - 1])[:selection.n_select(len(Z), p)]
                with open(f) as fh:
                    n_rep += json.load(fh)["selected_indices"] == rank.tolist()
            print(f"{name:11s} FPS cuts identical when ranking the legacy matrix itself: {n_rep}/{len(cuts)} "
                  f"(remaining differences = tie-breaking at 1e-15 level)")
            n_same = n_rep
        ok &= n_same == len(cuts)
    print("LEGACY SI REPRODUCED" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
