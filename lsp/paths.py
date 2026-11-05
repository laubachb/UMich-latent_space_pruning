"""Repository layout and local (machine-specific) tool paths.

Everything the pipeline writes goes under results/<dataset>/ and figures/<dataset>/;
everything it needs that is not tracked by git lives under external/ (see README).

Local executables (LAMMPS builds, the ChIMES fingerprint tool) are read from
configs/local_paths.json, which is gitignored; copy configs/local_paths.example.json
and edit it for your machine. Environment variables of the same name override it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
EXTERNAL = ROOT / "external"
MLEARN_DIR = EXTERNAL / "mlearn"          # external/mlearn/<El>/training.json
CHIMES_DIR = EXTERNAL / "chimes"          # external/chimes/<El>/{A.txt, train.xyzf, params.txt, ...}
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PLOT_DATA = ROOT / "plotting" / "data"
MTP_LEVELS = ROOT / "lsp" / "mtp_levels"

# Local tools: key -> (purpose, required by)
LOCAL_TOOL_KEYS = {
    "lmp_serial": "plain LAMMPS with the SNAP package (maml BispectrumCoefficients) — bispectrum",
    "lmp_chimes_fp": "LAMMPS built with chimesFF 'fingerprint' pair style — chimes_fingerprint",
    "chimes_fingerprint_tool": "fingerprint_all_alphas executable (cluster files -> histograms) — chimes_fingerprint",
}


def results_dir(dataset: str, *sub: str) -> Path:
    p = RESULTS / dataset
    for s in sub:
        p = p / s
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir(dataset: str) -> Path:
    p = FIGURES / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_data_dir(dataset: str) -> Path:
    p = PLOT_DATA / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p


def local_tool(key: str) -> str:
    """Path of a local executable, from $KEY (upper-cased) or configs/local_paths.json."""
    if key not in LOCAL_TOOL_KEYS:
        raise KeyError(f"unknown local tool '{key}'; known: {list(LOCAL_TOOL_KEYS)}")
    env = os.environ.get(key.upper())
    if env:
        return env
    f = CONFIGS / "local_paths.json"
    if f.exists():
        with open(f) as fh:
            val = json.load(fh).get(key)
        if val:
            return str(val)
    raise FileNotFoundError(
        f"local tool '{key}' ({LOCAL_TOOL_KEYS[key]}) is not configured: set ${key.upper()} "
        f"or add it to {f} (see configs/local_paths.example.json)"
    )
