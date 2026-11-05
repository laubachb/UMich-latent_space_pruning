#!/usr/bin/env python3
"""Check the frame-kernel QUESTS scoring against values from the direct kernel evaluation.

    python tests/test_quests_frame_kernel.py <dataset> <reference_quests.csv>

`reference_quests.csv` is a quests.csv written by the earlier implementation, which
re-evaluated quests.entropy.kernel_sum for every cut. Cuts are matched on
(method, percent, replicate) — the seed sequence is shared, so replicate r is the same
cut in both — and H, D and coverage must agree to float32 kernel precision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lsp import paths  # noqa: E402


def main(dataset: str, reference: str) -> int:
    old = pd.read_csv(reference)
    new = pd.read_csv(paths.results_dir(dataset, "analysis") / "quests.csv")
    j = old.merge(new, on=["method", "percent", "replicate"], suffixes=("_ref", "_new"))
    same_cut = j.n_atoms_ref == j.n_atoms_new
    print(f"[{dataset}] {len(j)} matched cuts ({len(old)} in reference); identical atom counts: {int(same_cut.sum())}")
    j = j[same_cut]
    ok = len(j) > 0
    for col, tol in (("H", 2e-4), ("D", 2e-4), ("coverage", 2e-3)):
        d = (j[f"{col}_new"] - j[f"{col}_ref"]).abs()
        print(f"  {col:9s} max |diff| {d.max():.2e}   mean {d.mean():.2e}   (tolerance {tol:.0e})")
        ok &= bool(d.max() <= tol)
    print("QUESTS FRAME KERNEL MATCHES DIRECT EVALUATION" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
