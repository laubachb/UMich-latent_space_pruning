#!/usr/bin/env python3
"""
Accumulated atom count vs dataset retention.

For each selection strategy (the four descriptor-space FPS methods plus the
random baseline), plot the total number of atoms contained in the pruned
dataset as a function of how much of the pool is retained. Because every
retention level has 10 replicate draws, each line is the per-replicate mean
total atom count and the shaded band is +/- 1 standard deviation across the
10 replicates.

All strategies converge to the full-pool atom count (~11,825) as retention
approaches 100%; they diverge most at aggressive pruning, where descriptors
that favor small structures (e.g. Bispectrum, SOAP) accumulate fewer atoms
than random selection at the same retention.

Run from the project root OR from the pruning/ directory:
    python pruning/atoms_vs_retention.py

Output: figures/atoms_vs_retention.png
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def _descriptor_pattern(name):
    return str(ROOT_DIR / 'descriptors' / name / 'replicates_structure_pruning_modified'
               / 'si_structures_*{percent}percent_replicate*.json')

METHOD_PATTERNS = {
    'Random': str(ROOT_DIR / 'random_for_ben'
                  / 'Training_si_structures_random_{percent}percent_replicate*.json'),
    'Behler': _descriptor_pattern('behler'),
    'Bispectrum': _descriptor_pattern('bispectrum'),
    'SOAP': _descriptor_pattern('soap'),
    'ChIMES': _descriptor_pattern('chimes'),
}

# Colorblind-safe (Okabe-Ito based) categorical palette; validated all-pairs.
# Random is the neutral reference (black, dashed). Each line also gets a
# distinct marker so identity never rests on color alone.
STYLE = {
    'Random':     dict(color='#000000', marker='o', linestyle='--'),
    'Behler':     dict(color='#E69F00', marker='s', linestyle='-'),
    'Bispectrum': dict(color='#0072B2', marker='^', linestyle='-'),
    'SOAP':       dict(color='#009E73', marker='D', linestyle='-'),
    'ChIMES':     dict(color='#D55E00', marker='v', linestyle='-'),
}


def atoms_per_replicate(pattern_template, percent):
    """Total atoms in each replicate dataset at a given retention percentage."""
    pattern = pattern_template.format(percent=percent)
    files = [f for f in glob.glob(pattern) if not f.endswith('_info.json')]
    totals = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        totals.append(sum(entry['num_atoms'] for entry in data))
    return np.array(totals)


def main():
    percentages = list(range(1, 10)) + list(range(10, 100, 10))

    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')

    stats = {}
    for method_name, pattern_template in METHOD_PATTERNS.items():
        means, stds = [], []
        for percent in percentages:
            totals = atoms_per_replicate(pattern_template, percent)
            means.append(totals.mean())
            stds.append(totals.std())
        means, stds = np.array(means), np.array(stds)
        stats[method_name] = (means, stds)

        style = STYLE[method_name]
        # Error bars are +/- 1 std of the total atom count across the 10 replicates.
        ax.errorbar(percentages, means, yerr=stds, label=method_name,
                    markersize=5, linewidth=2, markeredgecolor='white',
                    markeredgewidth=0.5, capsize=3, capthick=1, elinewidth=1,
                    **style)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dataset Retention (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of atoms in dataset', fontsize=12, fontweight='bold')
    # Match the normalized-composition figures: default log tick labels (10^0,
    # 10^1, 10^2) rather than explicit per-percentage ticks.
    ax.grid(True, which='both', axis='both', alpha=0.25, linewidth=0.6)
    ax.legend(title='Selection strategy', fontsize=10, title_fontsize=11,
              loc='upper left', frameon=True)

    output_path = FIGURES_DIR / 'atoms_vs_retention.png'
    plt.savefig(output_path, dpi=300)
    print(f'Saved: {output_path}')

    # Table at key retentions
    print('\n' + '=' * 70)
    print('MEAN TOTAL ATOMS PER DATASET (+/- std over 10 replicates)')
    print('=' * 70)
    key = [1, 5, 10, 20, 50, 90]
    print(f"{'method':<12}" + "".join(f"{str(p)+'%':<12}" for p in key))
    idx = {p: percentages.index(p) for p in key}
    for method_name, (means, stds) in stats.items():
        row = f"{method_name:<12}"
        for p in key:
            i = idx[p]
            row += f"{means[i]:.0f}±{stds[i]:.0f}".ljust(12)
        print(row)


if __name__ == '__main__':
    main()
