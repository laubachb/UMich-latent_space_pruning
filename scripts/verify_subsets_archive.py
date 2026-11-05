#!/usr/bin/env python3
"""Single-pass integrity check of a training-subset archive (scripts/export_training_subsets.py).

    python scripts/verify_subsets_archive.py results/lsp_training_subsets_YYYYMMDD.tar.gz

Streams the archive once and reports: number of subset files per element / sampler /
latent space, presence of README, manifest and full training sets, manifest-vs-archive
consistency, and that a random sample of subset files parses as JSON with the frame
count stated in its name. Exits non-zero on any inconsistency.
"""

from __future__ import annotations

import collections
import csv
import io
import json
import random
import shutil
import subprocess
import sys
import tarfile
import time


def main(path: str, sample: float = 0.002) -> int:
    t0 = time.time()
    proc = None
    if shutil.which("pigz"):
        proc = subprocess.Popen(["pigz", "-dc", path], stdout=subprocess.PIPE)
        tar = tarfile.open(fileobj=proc.stdout, mode="r|")
    else:
        tar = tarfile.open(path, "r|gz")
    per_space, names = collections.Counter(), set()
    manifest, readme, full, nbytes, checked, bad = None, False, 0, 0, 0, 0
    rng = random.Random(0)
    for m in tar:
        parts = m.name.split("/")
        nbytes += m.size
        if m.name.endswith("manifest.csv"):
            manifest = list(csv.DictReader(io.StringIO(tar.extractfile(m).read().decode())))
        elif m.name.endswith("README.md"):
            readme = True
        elif "_full_training_set" in m.name:
            full += 1
        else:
            per_space[(parts[1], parts[2], parts[3])] += 1
            names.add(m.name)
            if rng.random() < sample:
                frames = json.loads(tar.extractfile(m).read())
                n = int(m.name.split("frames_")[0].split("_")[-1])
                checked += 1
                bad += len(frames) != n or "structure" not in frames[0] or "outputs" not in frames[0]
    if proc is not None and proc.wait() != 0:
        print("decompression failed")
        return 1
    elements = sorted({k[0] for k in per_space})
    print(f"{len(names)} subset files, {full} full training sets, README {readme}, "
          f"{nbytes / 1e9:.1f} GB uncompressed ({time.time() - t0:.0f} s)")
    for e in elements:
        by_sampler = collections.Counter()
        for (el, s, _sp), n in per_space.items():
            if el == e:
                by_sampler[s] += n
        print(f"  {e}: {dict(by_sampler)}")
    print(f"  files per (element, sampler, space): min {min(per_space.values())}, max {max(per_space.values())}")
    ok = readme and manifest is not None and full == len(elements) and bad == 0
    if manifest is not None:
        present = all(r["file"] in names for r in manifest)
        counts = all(len(r["frame_indices"].split()) == int(r["n_frames"]) for r in manifest)
        print(f"  manifest rows {len(manifest)}; files present {present}; frame counts consistent {counts}")
        ok &= present and counts and len(manifest) == len(names)
    print(f"  sampled files parsed: {checked}, bad: {bad}")
    print("ARCHIVE OK" if ok else "ARCHIVE INCONSISTENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
