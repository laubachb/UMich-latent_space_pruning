"""
How much does each latent space's INFORMATION correspond to the PHYSICS?

The frames are identical across descriptor spaces, so any difference is a property
of the representation. For every cached latent space we reduce it to its
95%-variance PCA representation (denoise the many low-variance directions) and
measure how much each physical variable can be recovered from it:

  DECODABILITY - cross-validated skill of a Random-Forest probe predicting the
  physics from the latent coordinates: R^2 (continuous: E/atom, T, stress) or
  balanced accuracy (categorical: group, category). High = that representation
  encodes that physics; low = the physics is not legible in that latent space.

Output: figures/clustering_physics_encoding.png   (latent space x physics heatmap)
        clustering/cache/physics_encoding.csv

Run from this directory (after compute_descriptor_matrices.py):
    python physics_encoding.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import common

SEED = 42
PCA_VAR_RETAIN = 0.95
N_SPLITS = 5
MIN_VALID = 30

PHYS_CONTINUOUS = [
    ("energy_per_atom", "E/atom"),
    ("temperature_K", "T"),
    ("mean_normal_stress", "Stress"),
]
PHYS_CATEGORICAL = [
    ("group", "Group"),
    ("category", "Category"),
]


def reduced_representation(matrix):
    """Standardize, then project onto the 95%-variance PCA subspace."""
    X = common.standardize(matrix)
    n_keep = int(np.clip(PCA(n_components=PCA_VAR_RETAIN, random_state=SEED)
                         .fit(X).n_components_, 1, min(X.shape) - 1))
    return PCA(n_components=n_keep, random_state=SEED).fit_transform(X)


def participation_ratio(matrix):
    """Effective number of independent directions (for row ordering)."""
    X = common.standardize(matrix)
    lam = np.clip(PCA(n_components=min(X.shape)).fit(X).explained_variance_, 0, None)
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def score_continuous(Xr, y):
    """CV R^2 of an RF probe. Skill = max(R^2, 0)."""
    ok = np.isfinite(y)
    if ok.sum() < MIN_VALID or np.nanstd(y[ok]) == 0:
        return dict(probe=np.nan, skill=np.nan)
    rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1)
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    r2 = float(np.mean(cross_val_score(rf, Xr[ok], y[ok], scoring="r2", cv=cv)))
    return dict(probe=r2, skill=max(r2, 0.0))


def score_categorical(Xr, labels):
    """CV balanced accuracy of an RF probe. Skill above chance."""
    y = LabelEncoder().fit_transform(np.asarray(labels))
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < N_SPLITS:
        return dict(probe=np.nan, skill=np.nan)
    chance = 1.0 / len(classes)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    acc = float(np.mean(cross_val_score(rf, Xr, y, scoring="balanced_accuracy", cv=cv)))
    return dict(probe=acc, skill=max((acc - chance) / (1.0 - chance), 0.0))


def main():
    meta = pd.read_csv(common.CACHE_DIR / "metadata.csv")
    spaces = common.available_descriptors()
    if not spaces:
        raise SystemExit("No cached descriptors. Run compute_descriptor_matrices.py first.")
    print(f"Latent spaces: {spaces}")

    rows, pr = [], {}
    for name in spaces:
        Xr = reduced_representation(common.load_descriptor_matrix(name))
        pr[name] = participation_ratio(common.load_descriptor_matrix(name))
        print(f"  {name:11s} PR={pr[name]:.1f}  probing {Xr.shape[1]} PCs ...")
        for key, label in PHYS_CONTINUOUS:
            rows.append(dict(space=name, label=label,
                             **score_continuous(Xr, meta[key].to_numpy(dtype=float))))
        for key, label in PHYS_CATEGORICAL:
            rows.append(dict(space=name, label=label,
                             **score_categorical(Xr, meta[key].to_numpy())))

    df = pd.DataFrame(rows)
    df.to_csv(common.CACHE_DIR / "physics_encoding.csv", index=False)

    prop_labels = [lab for _, lab in PHYS_CONTINUOUS + PHYS_CATEGORICAL]
    order = sorted(spaces, key=lambda n: pr[n], reverse=True)   # most-informative first
    disp = [common.descriptor_label(n) for n in order]

    def grid(col):
        return np.array([[df[(df.space == n) & (df.label == lab)][col].iloc[0]
                          for lab in prop_labels] for n in order], dtype=float)

    skill, probe = grid("skill"), grid("probe")

    fig = plt.figure(figsize=(3, 4))
    # Manually placed square heatmap with a horizontal colorbar in a fixed slot
    # below, clear of the vertical x-labels (avoids tight_layout squishing the
    # grid). Square cells: axes width in inches == height in inches, i.e.
    # width_frac * 3 == height_frac * 4  ->  0.64*3 == 0.48*4 == 1.92 in.
    ax = fig.add_axes([0.30, 0.45, 0.64, 0.48])
    im = ax.imshow(skill, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(prop_labels)))
    ax.set_xticklabels(prop_labels, rotation=90, ha="center", fontsize=10)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(disp, fontsize=10)
    for i in range(skill.shape[0]):
        for j in range(skill.shape[1]):
            v = probe[i, j]
            txt = "n/a" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="w" if np.isfinite(skill[i, j]) and skill[i, j] < 0.5 else "k")
    cax = fig.add_axes([0.30, 0.16, 0.64, 0.028])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Decodability", fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    out = common.FIGURES_DIR / "clustering_physics_encoding.png"
    # Manual axes + fixed figsize => saved file is exactly 3.25x3.25 in.
    fig.savefig(out, dpi=250)
    print(f"Saved: {out}")

    print("\n" + "=" * 74)
    print("PHYSICAL DECODABILITY (R2 continuous / balanced-acc categorical); "
          "rows ordered by effective dimensionality")
    print("=" * 74)
    print(df.pivot(index="space", columns="label", values="probe")
          .reindex(order)[prop_labels].round(3).to_string())


if __name__ == "__main__":
    main()
