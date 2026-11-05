"""Training-set selection in every latent space, with several subsampling rules.

Samplers (all operate on the z-scored descriptor matrix of one latent space and
use the same replicate seeds everywhere):

    fps        farthest point sampling (skmatter, random first point). Greedy
               max-min: each pick is the frame farthest from the current set, so
               the cuts are NESTED (a cut at retention p is the first n_p frames of
               the ranking) and favour the extremes / outliers of the space.
    kmedoids   k-means with k = n_p clusters (seeded), one medoid per cluster (the
               frame nearest its centroid). Density-representative: picks typical
               frames of every populated region instead of its boundary. Not nested.
    leverage   randomized CUR on the frames: n_p frames drawn without replacement
               with probability proportional to their statistical leverage in the
               top-k principal subspace (k = components for 90 % of the variance).
               Stochastic, between random and FPS: it favours frames that are
               hard to express as combinations of others. Not nested.
    random     descriptor-free baseline (independent draws per retention).

A selection is identified as "<sampler>-<latent space>" ("random" for the
baseline) and stored in results/<dataset>/selection/<id>.json: FPS keeps the ten
rankings, the others store the frame indices of every (retention, replicate) cut.

Seeds: random.seed(42) -> N_REPLICATES replicate seeds (identical for every sampler, latent
space and dataset, as in the original Si scripts). Frames whose descriptor is
undefined (NaN rows) are excluded from that latent space's pool and recorded.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from . import data, descriptors, paths

PERCENTS = list(range(1, 10)) + list(range(10, 100, 10))     # 1..9, 10..90
N_REPLICATES = 25
SEED = 42
SAMPLERS = ["fps", "kmedoids", "leverage"]
DEFAULT_SAMPLER = "fps"


def replicate_seeds(n: int = N_REPLICATES, seed: int = SEED) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 2 ** 32 - 1) for _ in range(n)]


def n_select(n_frames: int, percent: int) -> int:
    return max(1, int(n_frames * percent / 100))


def method_id(sampler: str, descriptor: str | None = None) -> str:
    return "random" if sampler == "random" else f"{sampler}-{descriptor}"


def split_method(method: str) -> tuple[str, str | None]:
    if method == "random":
        return "random", None
    s, d = method.split("-", 1)
    return s, d


def selection_file(dataset: str, method: str) -> Path:
    return paths.results_dir(dataset, "selection") / f"{method}.json"


# ── samplers ────────────────────────────────────────────────────────────────
def fps_ranking(Z: np.ndarray, seed: int) -> np.ndarray:
    from skmatter.sample_selection import FPS
    fps = FPS(initialize="random", n_to_select=len(Z), random_state=seed)
    fps.fit(Z)
    return fps.selected_idx_


def kmedoids_cut(Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    if k >= len(Z):
        return np.arange(len(Z))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(Z)
    picks = []
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        if len(members) == 0:
            continue
        d = np.linalg.norm(Z[members] - km.cluster_centers_[c], axis=1)
        picks.append(members[np.argmin(d)])
    picks = np.array(sorted(set(picks)))
    if len(picks) < k:                      # empty clusters: top up with the farthest unpicked frames
        rest = np.setdiff1d(np.arange(len(Z)), picks)
        dmin = np.min(np.linalg.norm(Z[rest][:, None] - Z[picks][None], axis=2), axis=1)
        picks = np.concatenate([picks, rest[np.argsort(-dmin)[:k - len(picks)]]])
    return picks


def leverage_scores(Z: np.ndarray, var: float = 0.90) -> np.ndarray:
    Zc = Z - Z.mean(0)
    U, s, _ = np.linalg.svd(Zc, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), var) + 1)
    lev = np.sum(U[:, :k] ** 2, axis=1)
    return lev / lev.sum()


def leverage_cut(Z: np.ndarray, k: int, seed: int, p: np.ndarray | None = None) -> np.ndarray:
    p = leverage_scores(Z) if p is None else p
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(Z), size=min(k, len(Z)), replace=False, p=p))


# ── drivers ─────────────────────────────────────────────────────────────────
def _random_cuts(n: int, seeds: list[int], percents: list[int]) -> list[dict]:
    reps = []
    for s in seeds:
        cuts = {}
        for p in percents:
            np.random.seed(s)                       # one independent draw per (seed, retention)
            cuts[str(p)] = np.random.choice(n, n_select(n, p), replace=False).tolist()
        reps.append(dict(seed=int(s), cuts=cuts))
    return reps


def _kmedoids_rep(args):
    Z, pool, n, seed, percents = args
    return dict(seed=int(seed), cuts={str(p): pool[kmedoids_cut(Z, n_select(n, p), seed)].tolist() for p in percents})


def _compute(X: np.ndarray, sampler: str, seeds: list[int], percents: list[int]) -> tuple[str, list[dict], list[int]]:
    """(kind, replicates, excluded_frames) of one sampler on one raw descriptor matrix."""
    ok = np.isfinite(X).all(axis=1)
    pool = np.where(ok)[0]
    Z = descriptors.standardize(X[pool])
    n = len(X)
    if sampler == "fps":
        reps = [dict(seed=int(s), ranking=pool[fps_ranking(Z, s)].tolist()) for s in seeds]
        kind = "ranking"
    elif sampler == "kmedoids":
        from .descriptors.aggregate import parallel_frames
        reps = parallel_frames(_kmedoids_rep, [(Z, pool, n, s, percents) for s in seeds], "kmedoids replicates")
        kind = "cuts"
    elif sampler == "leverage":
        lev = leverage_scores(Z)
        reps = [dict(seed=int(s), cuts={str(p): pool[leverage_cut(Z, n_select(n, p), s, lev)].tolist() for p in percents})
                for s in seeds]
        kind = "cuts"
    else:
        raise KeyError(sampler)
    return kind, reps, np.where(~ok)[0].tolist()


def run(dataset: str, sampler: str, descriptor: str | None = None, force: bool = False,
        percents: list[int] = PERCENTS, n_replicates: int = N_REPLICATES, subdir: str = "selection") -> dict:
    """Compute (or load) the selection of one sampler in one latent space.

    The analysis set lives in results/<ds>/selection/ (PERCENTS x N_REPLICATES); other
    grids (e.g. the exported training subsets, 100 replicates x 7 retentions) go to
    another `subdir`. Seeds are the first n_replicates of the same seed sequence, so
    replicate r is the same cut wherever it appears."""
    method = method_id(sampler, descriptor)
    f = paths.results_dir(dataset, subdir) / f"{method}.json"
    if f.exists() and not force:
        with open(f) as fh:
            sel = json.load(fh)
        if sel["percents"] == list(percents) and len(sel["replicates"]) == n_replicates:
            return sel
    seeds = replicate_seeds(n_replicates)
    if sampler == "random":
        n = len(data.load_frames(dataset))
        sel = dict(dataset=dataset, method=method, sampler=sampler, descriptor=None, kind="cuts",
                   n_frames=n, n_features=0, percents=list(percents), excluded_frames=[],
                   replicates=_random_cuts(n, seeds, percents))
    else:
        X = descriptors.compute(dataset, descriptor)
        kind, reps, excluded = _compute(X, sampler, seeds, percents)
        sel = dict(dataset=dataset, method=method, sampler=sampler, descriptor=descriptor, kind=kind,
                   n_frames=int(X.shape[0]), n_features=int(X.shape[1]), percents=list(percents),
                   excluded_frames=excluded, replicates=reps)
    with open(f, "w") as fh:
        json.dump(sel, fh)
    print(f"[{dataset}] selection '{method}' ({len(seeds)} replicates, {len(percents)} retentions) -> {f}")
    return sel


def load(dataset: str, method: str, subdir: str = "selection") -> dict:
    with open(paths.results_dir(dataset, subdir) / f"{method}.json") as fh:
        return json.load(fh)


def iter_cuts(sel: dict):
    """Yield (percent, replicate_index, frame_indices) for every cut of a selection."""
    n = sel["n_frames"]
    for r, rep in enumerate(sel["replicates"]):
        for p in sel["percents"]:
            if sel["kind"] == "ranking":
                idx = np.asarray(rep["ranking"][:n_select(n, p)])
            else:
                idx = np.asarray(rep["cuts"][str(p)])
            yield p, r, idx


def available(dataset: str, sampler: str | None = None) -> list[str]:
    """Selection ids on disk, ordered random, then by sampler and descriptor order."""
    d = paths.results_dir(dataset, "selection")
    have = {p.stem for p in d.glob("*.json")}
    order = ["random"] if sampler in (None, "random") else []
    for s in ([sampler] if sampler and sampler != "random" else SAMPLERS):
        order += [method_id(s, n) for n in descriptors.available(dataset)]
    return [m for m in order if m in have]


def export_cuts(dataset: str, method: str, out_dir: Path) -> list[Path]:
    """Materialise every cut as a JSON list of frames (the format the model-fitting
    scripts consume): <out_dir>/<dataset>_<method>_<p>percent_replicate<r>.json."""
    from monty.serialization import dumpfn
    frames = data.load_frames(dataset)
    sel = load(dataset, method)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for p, r, idx in iter_cuts(sel):
        f = out_dir / f"{dataset}_{method}_{p}percent_replicate{r + 1:02d}.json"
        dumpfn([frames[i] for i in idx], f, indent=None)
        with open(f.with_name(f.stem + "_info.json"), "w") as fh:
            json.dump(dict(method=method, dataset=dataset, percent=p, replicate=r + 1,
                           seed=sel["replicates"][r]["seed"], n_original=sel["n_frames"],
                           n_selected=len(idx), selected_indices=idx.tolist()), fh)
        written.append(f)
    return written
