#!/usr/bin/env python3
"""
Normalized category composition plots across pruning rates — temperature-resolved.

Same as normalized_category_composition.py, but the defect-free bulk AIMD
snapshots are split out by simulation temperature instead of being lumped into
a single "Normal (300K)" bin plus an "Other" catch-all. This resolves the whole
dataset into 9 physically meaningful categories with no residual "Other":

    Strained            Si 2x2x2 cell strain/phonon-mode structures
    Surface             Si_surface_mp slabs
    Vacancy (300K)      vacancy AIMD @ 300 K
    Vacancy (3374K)     vacancy AIMD @ 3374 K
    Bulk (300K..3374K)  defect-free bulk AIMD along the 5-point temperature
                        ladder 300 / 843 / 1518 / 2530 / 3374 K

The lone "Ground state crystal" structure (no temperature tag) folds into the
T=0 endpoint of the bulk ladder, Bulk (300K).

Encoding is composite: hue = structure family, lightness = temperature within
the two sequences (Bulk = orange ramp, Vacancy = blue ramp). The categorical
stack order below was validated so every cross-family boundary stays
CVD-separable; the within-ramp steps are intentionally close (they carry a
sequential, not categorical, meaning) and are set apart by the white spacers.

Run from the project root OR from the pruning/ directory:
    python pruning/normalized_category_composition_by_temperature.py

Output: figures/normalized_category_composition_by_temperature.png
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

# Paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# Each method maps to a glob pattern with a {percent} placeholder. The random
# baseline lives in its own directory with a different filename prefix, so it
# needs its own pattern rather than sharing the descriptor layout.
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

# Bulk temperature ladder (defect-free AIMD), coolest -> hottest.
BULK_TEMPS = ['300', '843', '1518', '2530', '3374']

# Stack order (bottom -> top). Chosen so every cross-family boundary is
# CVD-separable: the strong blue|orange boundary (Vacancy 3374K | Bulk 3374K)
# sits in the middle to carry dichromats, and the off-axis green/magenta bands
# (Strained, Surface) are kept apart from the colors they would collide with.
CATEGORIES = [
    'Surface',
    'Vacancy (300K)', 'Vacancy (3374K)',
    'Bulk (3374K)', 'Bulk (2530K)', 'Bulk (1518K)', 'Bulk (843K)', 'Bulk (300K)',
    'Strained',
]

# Legend reading order: group by family, temperatures ascending.
LEGEND_ORDER = [
    'Strained', 'Surface',
    'Vacancy (300K)', 'Vacancy (3374K)',
    'Bulk (300K)', 'Bulk (843K)', 'Bulk (1518K)', 'Bulk (2530K)', 'Bulk (3374K)',
]

COLORS = {
    'Strained':        '#238B45',  # green
    'Surface':         '#DD3497',  # magenta
    'Vacancy (300K)':  '#9ECAE1',  # blue ramp, light
    'Vacancy (3374K)': '#2171B5',  # blue ramp, dark
    'Bulk (300K)':     '#FEE0B6',  # orange ramp, light (cool)
    'Bulk (843K)':     '#FDC171',
    'Bulk (1518K)':    '#FB9A3F',
    'Bulk (2530K)':    '#EC7014',
    'Bulk (3374K)':    '#C24E02',  # orange ramp, dark (hot)
}


def categorize_structure(description):
    desc = description.lower()
    if 'strain' in desc or 'mode' in desc:
        return 'Strained'
    if 'surface' in desc:
        return 'Surface'
    if 'vacancy' in desc:
        return 'Vacancy (3374K)' if '3374 K' in description else 'Vacancy (300K)'
    # Defect-free bulk AIMD: bin by simulation temperature.
    for temp in BULK_TEMPS:
        if f'{temp} K' in description:
            return f'Bulk ({temp}K)'
    # 'Ground state crystal' has no temperature tag -> T=0 endpoint of the ladder.
    return 'Bulk (300K)'


def analyze_method_at_percentage(pattern_template, percent):
    pattern = pattern_template.format(percent=percent)
    files = [f for f in glob.glob(pattern) if not f.endswith('_info.json')]

    category_total = defaultdict(int)
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        for entry in data:
            cat = categorize_structure(entry['description'])
            category_total[cat] += 1

    return dict(category_total)


def main():
    percentages = list(range(1, 10)) + list(range(10, 100, 10))

    all_data = {}
    for method_name, pattern_template in METHOD_PATTERNS.items():
        method_data = {}
        for percent in percentages:
            method_data[percent] = analyze_method_at_percentage(pattern_template, percent)
        all_data[method_name] = method_data

    method_names = list(METHOD_PATTERNS.keys())
    # 2x3 grid holds the random baseline plus the four descriptors; the unused
    # sixth cell hosts the shared legend. Constrained layout reserves room for
    # the shared x/y labels and subplot titles so nothing overlaps.
    n_cols = 3
    fig, axes = plt.subplots(2, n_cols, figsize=(9, 4.5), squeeze=False, layout='constrained')

    poly_by_category = {}
    for idx, method_name in enumerate(method_names):
        ax = axes[idx // n_cols][idx % n_cols]
        x = list(percentages)

        totals = [sum(all_data[method_name][p].values()) for p in percentages]
        data_by_category = {}
        for cat in CATEGORIES:
            normalized = []
            for i, p in enumerate(percentages):
                count = all_data[method_name][p].get(cat, 0)
                normalized.append(100 * count / totals[i] if totals[i] > 0 else 0)
            data_by_category[cat] = normalized

        polys = ax.stackplot(x, *[data_by_category[cat] for cat in CATEGORIES],
                             labels=CATEGORIES,
                             colors=[COLORS[cat] for cat in CATEGORIES],
                             alpha=0.85, edgecolor='white', linewidth=1.0)
        if not poly_by_category:
            poly_by_category = dict(zip(CATEGORIES, polys))

        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.grid(False)
        ax.set_xticks(x)
        ax.set_ylim([0, 100])
        ax.set_xscale('log')

    # Hide any unused cells and put the shared legend in the last empty one.
    for idx in range(len(method_names), axes.size):
        axes[idx // n_cols][idx % n_cols].axis('off')

    # One shared x label, one shared y label, one legend in the empty cell,
    # ordered by family (temperatures ascending) rather than by stack order.
    fig.supxlabel('Pruning Percentage (%)', fontsize=12, fontweight='bold')
    fig.supylabel('Composition (%)', fontsize=12, fontweight='bold')
    legend_ax = axes[len(method_names) // n_cols][len(method_names) % n_cols]
    legend_ax.legend([poly_by_category[c] for c in LEGEND_ORDER], LEGEND_ORDER,
                     fontsize=8.5, loc='center', frameon=False,
                     title='Category', title_fontsize=10)

    output_path = FIGURES_DIR / 'normalized_category_composition_by_temperature.png'
    plt.savefig(output_path, dpi=300)
    print(f'Saved: {output_path}')

    # Print table at key percentages
    print('\n' + '=' * 110)
    print('NORMALIZED CATEGORY COMPOSITION (%) AT KEY PRUNING RATES')
    print('=' * 110)
    key_percentages = [1, 5, 10, 20, 50, 90]
    header = f"{'%':<6}" + "".join(f"{c:<16}" for c in LEGEND_ORDER)

    for method_name in method_names:
        print(f'\n{method_name}:')
        print('-' * 110)
        print(header)
        print('-' * 110)
        for percent in key_percentages:
            counts = all_data[method_name].get(percent, {})
            total = sum(counts.values())
            if total == 0:
                print(f"{percent:<6} No data")
                continue
            row = f"{percent:<6}"
            for cat in LEGEND_ORDER:
                row += f"{100 * counts.get(cat, 0) / total:<16.1f}"
            print(row)


if __name__ == '__main__':
    main()
