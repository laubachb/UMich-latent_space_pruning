#!/usr/bin/env python3
"""
Analyze how category selections change with increasing pruning rates (1-10%).
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def categorize_structure(description):
    """Categorize a structure based on its description."""
    if 'strain' in description.lower() or 'mode' in description.lower():
        return 'Strained'
    elif 'vacancy' in description.lower():
        if '3374 K' in description:
            return 'High Temp (3374K)'
        else:
            return 'Vacancy (300K)'
    elif 'surface' in description.lower():
        return 'Surface'
    elif '3374 K' in description:
        return 'High Temp (3374K)'
    elif '300 K' in description:
        return 'Normal (300K)'
    else:
        return 'Other'


def analyze_method_at_percentage(method_dir, percent):
    """Analyze all replicates for a method at a given percentage."""
    files = glob.glob(f'{method_dir}/replicates_structure_pruning_modified/si_structures_mean_std_{percent}percent_replicate*.json')
    files = [f for f in files if not f.endswith('_info.json')]

    category_total = defaultdict(int)

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)

        for entry in data:
            cat = categorize_structure(entry['description'])
            category_total[cat] += 1

    return dict(category_total)


def main():
    # Define categories and colors
    categories = ['Strained', 'High Temp (3374K)', 'Normal (300K)', 'Vacancy (300K)', 'Surface', 'Other']
    colors = {
        'Strained': '#E63946',
        'High Temp (3374K)': '#F77F00',
        'Normal (300K)': '#06A77D',
        'Vacancy (300K)': '#457B9D',
        'Surface': '#A23B72',
        'Other': '#999999'
    }

    methods = {
        'behler': 'Behler',
        'bispectrum': 'Bispectrum',
        'pruning': 'SOAP'
    }

    percentages = range(1, 11)

    # Collect data for all methods and percentages
    all_data = {}

    for method_dir, method_name in methods.items():
        method_data = {}
        for percent in percentages:
            category_counts = analyze_method_at_percentage(method_dir, percent)
            method_data[percent] = category_counts
        all_data[method_name] = method_data

    # Print summary
    print('='*100)
    print('CATEGORY TOTALS ACROSS ALL REPLICATES BY PRUNING PERCENTAGE')
    print('='*100)

    for method_name in ['Behler', 'Bispectrum', 'SOAP']:
        print(f'\n{method_name}:')
        print('-' * 100)
        print(f"{'%':<5} {'Strained':<12} {'High Temp':<12} {'Normal':<12} {'Vacancy':<12} {'Surface':<12} {'Other':<12} {'Total':<8}")
        print('-' * 100)

        for percent in percentages:
            counts = all_data[method_name][percent]
            total = sum(counts.values())

            print(f"{percent:<5} "
                  f"{counts.get('Strained', 0):<12} "
                  f"{counts.get('High Temp (3374K)', 0):<12} "
                  f"{counts.get('Normal (300K)', 0):<12} "
                  f"{counts.get('Vacancy (300K)', 0):<12} "
                  f"{counts.get('Surface', 0):<12} "
                  f"{counts.get('Other', 0):<12} "
                  f"{total:<8}")

    # Create visualization
    fig = plt.figure(figsize=(18, 10))

    method_names = ['Behler', 'Bispectrum', 'SOAP']

    # 1. Line plots showing trends for each method
    for idx, method_name in enumerate(method_names):
        ax = plt.subplot(2, 3, idx + 1)

        x = list(percentages)

        for cat in categories:
            if cat == 'Other':
                continue  # Skip 'Other' for cleaner visualization

            counts = [all_data[method_name][p].get(cat, 0) for p in percentages]

            # Only plot if there's data
            if max(counts) > 0:
                ax.plot(x, counts, marker='o', linewidth=2.5, markersize=8,
                       label=cat, color=colors[cat])

        ax.set_xlabel('Pruning Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Total Selections (across 10 replicates)', fontsize=11, fontweight='bold')
        ax.set_title(f'{method_name} - Category Trends', fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(x)

    # 2. Stacked area plots
    for idx, method_name in enumerate(method_names):
        ax = plt.subplot(2, 3, idx + 4)

        x = list(percentages)

        # Prepare data for stacking
        data_by_category = {}
        for cat in categories:
            if cat == 'Other':
                continue
            data_by_category[cat] = [all_data[method_name][p].get(cat, 0) for p in percentages]

        # Stack them
        ax.stackplot(x, *[data_by_category[cat] for cat in categories if cat != 'Other'],
                     labels=[cat for cat in categories if cat != 'Other'],
                     colors=[colors[cat] for cat in categories if cat != 'Other'],
                     alpha=0.8, edgecolor='white', linewidth=1.5)

        ax.set_xlabel('Pruning Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Total Selections (stacked)', fontsize=11, fontweight='bold')
        ax.set_title(f'{method_name} - Category Composition', fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(x)

    plt.suptitle('Category Selection Trends with Increasing Pruning Rate (1-10%)\n' +
                 'Total selections across all 10 replicates per percentage',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = '/Users/blaubach/claude_stuff/category_trends_by_pruning.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\n\nVisualization saved to: {output_path}')

    # Create a second figure with all methods on same plots for comparison
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))

    # For each category, plot all three methods
    cat_subplot = {
        'Strained': (0, 0),
        'High Temp (3374K)': (0, 1),
        'Normal (300K)': (0, 2),
        'Vacancy (300K)': (1, 0),
        'Surface': (1, 1),
    }

    method_colors_for_comparison = {
        'Behler': '#2E86AB',
        'Bispectrum': '#A23B72',
        'SOAP': '#F18F01'
    }

    for cat, (row, col) in cat_subplot.items():
        ax = axes[row, col]

        x = list(percentages)

        for method_name in method_names:
            counts = [all_data[method_name][p].get(cat, 0) for p in percentages]
            ax.plot(x, counts, marker='o', linewidth=2.5, markersize=8,
                   label=method_name, color=method_colors_for_comparison[method_name])

        ax.set_xlabel('Pruning Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Total Selections', fontsize=11, fontweight='bold')
        ax.set_title(f'{cat}', fontsize=12, fontweight='bold', pad=10,
                     color=colors[cat])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(x)
        ax.set_ylim(bottom=0)

    # Last subplot: Total comparison
    ax = axes[1, 2]
    x = list(percentages)

    for method_name in method_names:
        totals = [sum(all_data[method_name][p].values()) for p in percentages]
        ax.plot(x, totals, marker='o', linewidth=2.5, markersize=8,
               label=method_name, color=method_colors_for_comparison[method_name])

    ax.set_xlabel('Pruning Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Selections', fontsize=11, fontweight='bold')
    ax.set_title('Total Selections (All Categories)', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(x)

    plt.suptitle('Method Comparison: Category Trends Across Pruning Rates\n' +
                 'Comparing how each method selects different structure types',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path2 = '/Users/blaubach/claude_stuff/category_comparison_by_pruning.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f'Comparison visualization saved to: {output_path2}')


if __name__ == '__main__':
    main()
