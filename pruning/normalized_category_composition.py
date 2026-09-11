#!/usr/bin/env python3
"""
Normalized category composition plots across pruning rates.

Shows how each descriptor's FPS selects from different structure categories
(strained, high-temp, surface, etc.) as a function of pruning percentage.

Run from the project root OR from the analysis/ directory:
    python analysis/normalized_category_composition.py
    cd analysis && python normalized_category_composition.py

Output: figures/normalized_category_composition.png
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

CATEGORIES = ['Strained', 'High Temp (3374K)', 'Normal (300K)', 'Vacancy (300K)', 'Surface', 'Other']
COLORS = {
    'Strained': '#E63946',
    'High Temp (3374K)': '#F77F00',
    'Normal (300K)': '#06A77D',
    'Vacancy (300K)': '#457B9D',
    'Surface': '#A23B72',
    'Other': '#999999',
}


def categorize_structure(description):
    desc = description.lower()
    if 'strain' in desc or 'mode' in desc:
        return 'Strained'
    elif 'vacancy' in desc:
        return 'High Temp (3374K)' if '3374 K' in description else 'Vacancy (300K)'
    elif 'surface' in desc:
        return 'Surface'
    elif '3374 K' in description:
        return 'High Temp (3374K)'
    elif '300 K' in description:
        return 'Normal (300K)'
    return 'Other'


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

    legend_handles = None
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
                             alpha=0.8, edgecolor='white', linewidth=1.5)
        if legend_handles is None:
            legend_handles = polys

        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.grid(False)
        ax.set_xticks(x)
        ax.set_ylim([0, 100])
        ax.set_xscale('log')

    # Hide any unused cells and put the shared legend in the last empty one.
    for idx in range(len(method_names), axes.size):
        axes[idx // n_cols][idx % n_cols].axis('off')

    # One shared x label, one shared y label, one legend in the empty cell.
    fig.supxlabel('Pruning Percentage (%)', fontsize=12, fontweight='bold')
    fig.supylabel('Composition (%)', fontsize=12, fontweight='bold')
    legend_ax = axes[(len(method_names)) // n_cols][(len(method_names)) % n_cols]
    legend_ax.legend(legend_handles, CATEGORIES, fontsize=10, loc='center',
                     frameon=False, title='Category', title_fontsize=11)

    output_path = FIGURES_DIR / 'normalized_category_composition.png'
    plt.savefig(output_path, dpi=300)
    print(f'Saved: {output_path}')

    # Print table at key percentages
    print('\n' + '='*100)
    print('NORMALIZED CATEGORY COMPOSITION (%) AT KEY PRUNING RATES')
    print('='*100)
    key_percentages = [1, 5, 10, 20, 50, 90]
    header = f"{'%':<6} {'Strained':<12} {'High Temp':<12} {'Normal':<12} {'Vacancy':<12} {'Surface':<12} {'Other':<12}"

    for method_name in method_names:
        print(f'\n{method_name}:')
        print('-' * 100)
        print(header)
        print('-' * 100)
        for percent in key_percentages:
            counts = all_data[method_name].get(percent, {})
            total = sum(counts.values())
            if total == 0:
                print(f"{percent:<6} No data")
                continue
            print(
                f"{percent:<6} "
                f"{100*counts.get('Strained', 0)/total:<12.1f}"
                f"{100*counts.get('High Temp (3374K)', 0)/total:<12.1f}"
                f"{100*counts.get('Normal (300K)', 0)/total:<12.1f}"
                f"{100*counts.get('Vacancy (300K)', 0)/total:<12.1f}"
                f"{100*counts.get('Surface', 0)/total:<12.1f}"
                f"{100*counts.get('Other', 0)/total:<12.1f}"
            )


if __name__ == '__main__':
    main()
