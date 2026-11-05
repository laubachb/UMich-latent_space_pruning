"""Full-frame ChIMES cluster-graph fingerprint (Laubach et al.).

A structure-level descriptor that does not go through per-atom averaging: every
2-, 3- and 4-body cluster inside the ChIMES cutoffs is a point in the space of
Morse-transformed cluster edge lengths; the pairwise distances between all
clusters of the same body order are histogrammed (100 bins over the normalised
distance range) and the three normalised histograms (2b, 3b, 4b; 300 values) are
the frame's fingerprint. At alpha = 1 (the setting used here; single-element
systems have no compositional term) the fingerprint is purely structural.

Pipeline (both tools are external, see configs/local_paths.json):
  1. LAMMPS with `pair_style chimesFF fingerprint 1` dumps the cluster edge lists
     of each frame (<id>.all-{2,3,4}b-clusters.txt), using the SAME ChIMES params
     file (cutoffs, S_MINIM, Morse lambda) as the A-matrix descriptor of the dataset.
  2. `fingerprint_all_alphas <params> <cluster_dir> <ids...>` prints the histograms;
     the alpha = 1.00 "centrality" lines are kept (at alpha = 1 the two
     composition-ranking variants coincide).

Frames with fewer than two clusters in some body order have no fingerprint (NaN).
Cluster files are written to node-local scratch (TMPDIR) and only the histograms are
kept, so no per-frame files land on the shared file system.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase.io import write

from .. import paths
from ..data import to_ase

MASSES = {"Si": 28.0855, "Cu": 63.546, "Ge": 72.63, "Li": 6.941, "Mo": 95.95, "Ni": 58.6934}

LAMMPS_INPUT = """units real
newton on
atom_style atomic
atom_modify sort 0 0.0
atom_modify map array
neighbor 1.0 bin
neigh_modify delay 0 every 1 check yes
read_data frame.data.in
mass 1 {mass}
pair_style chimesFF fingerprint 1
pair_coeff * * {params}
thermo_style custom step pe
run 0
"""


def dump_clusters(atoms, element: str, params_file: Path, out_dir: Path, fid: int, lmp: str):
    work = Path(tempfile.mkdtemp(prefix=f"fp{fid}_", dir=os.environ.get("TMPDIR", "/tmp")))
    try:
        write(work / "frame.data.in", atoms, format="lammps-data", specorder=[element],
              atom_style="atomic")
        (work / "fp.in").write_text(LAMMPS_INPUT.format(mass=MASSES[element], params=params_file))
        env = dict(os.environ, MV2_ENABLE_AFFINITY="0")
        r = subprocess.run([lmp, "-i", "fp.in"], cwd=work, env=env, capture_output=True,
                           text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"LAMMPS fingerprint run failed for frame {fid}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        for b in "234":
            parts = sorted(f for f in os.listdir(work) if f.endswith(f".{b}b_clusters.txt"))
            with open(out_dir / f"{fid}.all-{b}b-clusters.txt", "w") as o:
                for f in parts:
                    o.write((work / f).read_text())
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _dump_one(args):
    dump_clusters(*args)


def run_fingerprint_tool(params_file: Path, cluster_dir: Path, ids: list[int], tool: str,
                         alpha: float = 1.0) -> dict[int, np.ndarray]:
    env = dict(os.environ, MV2_ENABLE_AFFINITY="0")
    r = subprocess.run([tool, str(params_file), str(cluster_dir), *map(str, ids)], cwd=cluster_dir,
                       env=env, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError(f"fingerprint tool failed:\n{r.stderr[-2000:]}")
    hists: dict[int, dict[str, np.ndarray]] = {}
    for line in r.stdout.splitlines():
        t = line.split()
        if len(t) != 104 or t[1] != "centrality" or abs(float(t[2]) - alpha) > 1e-9:
            continue
        hists.setdefault(int(t[0]), {})[t[3]] = np.array(t[4:], dtype=float)
    out = {}
    for fid in ids:
        h = hists.get(fid, {})
        vec = np.full(300, np.nan)
        for k, b in enumerate(("2b", "3b", "4b")):
            if b in h and np.isfinite(h[b]).all() and h[b].sum() > 0:
                vec[100 * k:100 * (k + 1)] = h[b]
        out[fid] = vec
    return out


def structure_descriptors(data, params: dict, element: str, chimes_dir: str | Path,
                          chunk: int = 50) -> np.ndarray:
    lmp = paths.local_tool("lmp_chimes_fp")
    tool = paths.local_tool("chimes_fingerprint_tool")
    params_file = Path(chimes_dir) / params.get("params_file", "params.txt")
    if not params_file.exists():
        raise FileNotFoundError(params_file)
    alpha = float(params.get("alpha", 1.0))
    cluster_dir = Path(tempfile.mkdtemp(prefix="chimes_fp_", dir=os.environ.get("TMPDIR", "/tmp")))
    X = np.full((len(data), 300), np.nan)
    try:
        from .aggregate import parallel_frames
        parallel_frames(_dump_one, [(to_ase(d["structure"]), element, params_file, cluster_dir, i, lmp)
                                    for i, d in enumerate(data)], "chimes fingerprint clusters")
        ids = list(range(len(data)))
        for c0 in range(0, len(ids), chunk):
            print(f"  chimes fingerprint histograms {c0}/{len(ids)}", end="\r", flush=True)
            for fid, vec in run_fingerprint_tool(params_file, cluster_dir, ids[c0:c0 + chunk], tool, alpha).items():
                X[fid] = vec
        print()
    finally:
        shutil.rmtree(cluster_dir, ignore_errors=True)
    bad = np.where(~np.isfinite(X).all(axis=1))[0]
    if len(bad):
        print(f"  WARNING: {len(bad)} frames without a complete fingerprint: {bad.tolist()}")
    return X
