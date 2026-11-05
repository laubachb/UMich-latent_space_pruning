#!/usr/bin/env python3
"""Export the selected training subsets of every dataset as ready-to-fit JSON files in one tar.

    python scripts/export_training_subsets.py --out /path/lsp_training_subsets.tar.gz
    python scripts/export_training_subsets.py Si Cu --replicates 100 --percents 1 2 5 10 25 50 75

For every dataset x sampler (fps, kmedoids, leverage, random) x latent space the
selections are computed (or loaded) with the requested number of replicates and
retention levels (results/<ds>/selection_export/<method>.json; the same seed sequence
as the analysis, so replicate r <= 10 is exactly the cut of the figures), and each cut
is written as a JSON list of frames in the mlearn training.json format.

Nothing is written to the file system except the tar itself: every subset is
serialised in memory and streamed into the archive (tens of thousands of files).

Archive layout (see the README.md inside the tar):
    lsp_training_subsets/README.md
    lsp_training_subsets/manifest.csv
    lsp_training_subsets/<El>/<El>_full_training_set.json
    lsp_training_subsets/<El>/<sampler>/<space>/<El>_trainsubset_<sampler>_<space>_retain<PPP>pct_<NNN>frames_rep<RRR>.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lsp import config, data, descriptors, paths, selection  # noqa: E402

EXPORT_PERCENTS = [1, 2, 5, 10, 25, 50, 75]
EXPORT_REPLICATES = 100
SAMPLERS = ["fps", "kmedoids", "leverage", "random"]
# latent spaces exported: per-atom mean of the atom-centred descriptors + the structure-level ones
SPACES = ["soap_mean", "behler_mean", "bispectrum_mean", "mtp_mean", "chimes", "chimes_fingerprint", "euler"]
SPACE_NAMES = {
    "soap_mean": "soap-peratom-mean", "behler_mean": "behler-parrinello-peratom-mean",
    "bispectrum_mean": "bispectrum-peratom-mean", "mtp_mean": "mtp-peratom-mean",
    "chimes": "chimes-amatrix-peratom", "chimes_fingerprint": "chimes-fingerprint-fullframe",
    "euler": "euler-characteristic-fullframe",
}
SAMPLER_TEXT = {
    "fps": "farthest point sampling in the latent space (greedy max-min, random first point; cuts are nested)",
    "kmedoids": "k-means with k = subset size on the latent space, one medoid (frame nearest its centroid) per cluster",
    "leverage": "randomized CUR: frames drawn without replacement with probability proportional to their statistical "
                "leverage in the 90 %-variance principal subspace of the latent space",
    "random": "uniform random draw without replacement (descriptor-free baseline)",
}
SPACE_TEXT = {
    "soap_mean": "SOAP power spectrum (dscribe; GAP settings of Zuo et al.), per-atom mean",
    "behler_mean": "Behler-Parrinello G2+G4 symmetry functions (NNP settings of Zuo et al.), per-atom mean",
    "bispectrum_mean": "bispectrum coefficients (SNAP settings of Zuo et al.), per-atom mean",
    "mtp_mean": "MTP basis functions (level and cutoff of Zuo et al., fixed Chebyshev radial basis), per-atom mean",
    "chimes": "ChIMES A-matrix energy row / N of T. Sundberg's per-element ChIMES fit (per-atom cluster-polynomial sums)",
    "chimes_fingerprint": "full-frame ChIMES cluster-graph fingerprint (alpha = 1; 2b/3b/4b histograms) with the same ChIMES params",
    "euler": "Euler characteristic curve chi(r)/N of the periodic Rips filtration, 0-6 A, 64 radii (topological reference)",
}


def readme(datasets, percents, n_rep):
    lines = [
        "# Training subsets for descriptor-induced sampling bias in MLIPs",
        "", f"Generated {date.today().isoformat()} from the `lsp` package "
        "(https://github.com/laubachb/UMich-latent_space_pruning, branch laubachb/multi-dataset).", "",
        "## What is in here",
        "", "For each element's benchmark training set (Zuo et al., J. Phys. Chem. A 124, 731 (2020); "
        "mlearn data/<El>/training.json, DFT energies/forces/stresses) the frames were ranked or drawn in several "
        "descriptor latent spaces by several subsampling rules, and every resulting subset is written out as a "
        "ready-to-fit JSON file. Train an MLIP on a subset and test on the element's mlearn test set "
        "(data/<El>/test.json, not included) or on the frames of the full training set that the subset leaves out.", "",
        f"Elements: {', '.join(datasets)}.",
        f"Retention levels: {', '.join(str(p) + ' %' for p in percents)} of the full training set "
        "(subset size = floor(retention x n_frames), at least 1).",
        f"Replicates: {n_rep} per (element, sampler, latent space, retention); replicate r uses seed r of one fixed "
        "seed sequence (random.seed(42)) everywhere, and replicates 1-10 are exactly the cuts analysed in the study's figures.", "",
        "## File names", "",
        "    <El>/<sampler>/<space>/<El>_trainsubset_<sampler>_<space>_retain<PPP>pct_<NNN>frames_rep<RRR>.json", "",
        "    <El>      element (Si, Cu, Ge, Li, Mo, Ni)",
        "    <sampler> how the frames were chosen:",
    ] + [f"              {k:9s} {v}" for k, v in SAMPLER_TEXT.items()] + [
        "    <space>   the latent space the sampler worked in (`nodescriptor` for the random baseline):",
    ] + [f"              {SPACE_NAMES[k]:34s} {v}" for k, v in SPACE_TEXT.items()] + [
        "    <PPP>     retention in percent of the full training set (zero-padded)",
        "    <NNN>     number of frames in the file (zero-padded)",
        "    <RRR>     replicate number 001-" + f"{n_rep:03d}", "",
        "Example: Cu/fps/soap-peratom-mean/Cu_trainsubset_fps_soap-peratom-mean_retain010pct_026frames_rep003.json",
        "is the third FPS replicate in the SOAP (per-atom mean) latent space keeping 10 % (26 frames) of the Cu training set.", "",
        "## File format", "",
        "Each file is a JSON list of frames in the mlearn format (monty-serialisable): `structure` (pymatgen Structure dict), "
        "`outputs` {`energy` eV (total), `forces` eV/A (n_atoms x 3), `virial_stress` (6, kbar, Voigt)}, `description`, "
        "`group` (AIMD-NVT / Vacancy / Elastic / Surface), `num_atoms`, `element`, `tag`. Load with "
        "`monty.serialization.loadfn` (or `json.load` + `pymatgen.core.Structure.from_dict`). "
        "`<El>/<El>_full_training_set.json` is the complete training set in the same format; `manifest.csv` lists every "
        "subset with its frame indices into that full set, so held-out frames are `all - indices`.", "",
        "## Notes", "",
        "* Per-atom mean = the descriptor's linear design row divided by the atom count, i.e. what a linear or "
        "average-kernel model (SNAP, MTP, ChIMES, GAP) sees for a frame.",
        "* Li's ChIMES fit has no 4-body term, so its fingerprint is 2b+3b only, and 22 very small Li cells have too few "
        "3-body clusters to be fingerprinted; a few Mo cells likewise. Such frames are never selected in that one latent "
        "space (listed in manifest.csv column `excluded_frames_in_space`).",
        "* Descriptor hyperparameters per element are in the repository's configs/<El>.json.",
    ]
    return "\n".join(lines) + "\n"


def frame_json_cache(dataset):
    """Compact JSON text of every frame of the dataset (serialised once)."""
    from monty.json import jsanitize
    frames = data.load_frames(dataset)
    return [json.dumps(jsanitize(d, strict=True), separators=(",", ":")) for d in frames], frames


def add_text(tar, name, text, mtime):
    b = text.encode()
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(b), mtime, 0o644
    tar.addfile(info, io.BytesIO(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", default=[d for d in config.available_datasets() if d != "si_legacy"])
    ap.add_argument("--percents", nargs="+", type=int, default=EXPORT_PERCENTS)
    ap.add_argument("--replicates", type=int, default=EXPORT_REPLICATES)
    ap.add_argument("--samplers", nargs="+", default=SAMPLERS)
    ap.add_argument("--spaces", nargs="+", default=SPACES)
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "lsp_training_subsets.tar.gz")
    ap.add_argument("--selections-only", action="store_true", help="compute the export selections, write no tar")
    a = ap.parse_args()
    root = "lsp_training_subsets"
    mtime = int(time.time())
    t0 = time.time()

    # 1. selections (cached in results/<ds>/selection_export/)
    sels = {}
    for ds in a.datasets:
        for s in a.samplers:
            spaces = [None] if s == "random" else [sp for sp in a.spaces if sp in descriptors.available(ds)]
            for sp in spaces:
                sels[(ds, s, sp)] = selection.run(ds, s, sp, percents=a.percents, n_replicates=a.replicates,
                                                  subdir="selection_export")
    print(f"selections ready ({time.time() - t0:.0f} s)", flush=True)
    if a.selections_only:
        return

    # 2. stream everything into the tar
    a.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = io.StringIO()
    mw = csv.writer(manifest)
    mw.writerow(["file", "element", "sampler", "latent_space", "latent_space_id", "retention_pct", "replicate", "seed",
                 "n_frames", "n_atoms", "frame_indices", "excluded_frames_in_space"])
    n_files = 0
    with tarfile.open(a.out, "w:gz", compresslevel=6) as tar:
        add_text(tar, f"{root}/README.md", readme(a.datasets, a.percents, a.replicates), mtime)
        for ds in a.datasets:
            el = config.load(ds)["element"]
            texts, frames = frame_json_cache(ds)
            natoms = [len(d["structure"]) for d in frames]
            add_text(tar, f"{root}/{el}/{el}_full_training_set.json", "[" + ",".join(texts) + "]", mtime)
            for (d2, s, sp), sel in sels.items():
                if d2 != ds:
                    continue
                space = "nodescriptor" if sp is None else SPACE_NAMES[sp]
                excluded = " ".join(map(str, sel["excluded_frames"]))
                for p, r, idx in selection.iter_cuts(sel):
                    idx = [int(i) for i in idx]
                    name = f"{el}_trainsubset_{s}_{space}_retain{p:03d}pct_{len(idx):03d}frames_rep{r + 1:03d}.json"
                    path = f"{root}/{el}/{s}/{space}/{name}"
                    add_text(tar, path, "[" + ",".join(texts[i] for i in idx) + "]", mtime)
                    mw.writerow([path, el, s, space, sp or "", p, r + 1, sel["replicates"][r]["seed"], len(idx),
                                 sum(natoms[i] for i in idx), " ".join(map(str, idx)), excluded])
                    n_files += 1
                print(f"  {el} {s:9s} {space:34s} {n_files:6d} files ({time.time() - t0:.0f} s)", flush=True)
        add_text(tar, f"{root}/manifest.csv", manifest.getvalue(), mtime)
    print(f"wrote {n_files} subsets -> {a.out} ({a.out.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
