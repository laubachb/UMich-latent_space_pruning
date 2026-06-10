#!/usr/bin/env python3
"""
Analyze structure categories at 1% pruning for each method and replicate.
Show histogram of category counts across replicates.
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


def analyze_replicate(json_file):
    """Analyze a single replicate file and return category counts."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    category_counts = defaultdict(int)
    for entry in data:
        cat = categorize_structure(entry['description'])
        category_counts[cat] += 1

    return dict(category_counts)


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

    # Collect data for each method
    methods = {
        'behler': 'Behler',
        'bispectrum': 'Bispectrum',
        'pruning': 'SOAP'
    }

    all_data = {}

    for method_dir, method_name in methods.items():
        files = glob.glob(f'{method_dir}/replicates_structure_pruning_modified/si_structures_mean_std_1percent_replicate*.json')
        files = [f for f in sorted(files) if not f.endswith('_info.json')]

        replicate_data = []

        for file in files:
            replicate_num = file.split('replicate')[-1].split('.json')[0]
            category_counts = analyze_replicate(file)
            replicate_data.append({
                'replicate': replicate_num,
                'counts': category_counts,
                'file': file
            })

        all_data[method_name] = replicate_data

    # Print detailed breakdown
    print('='*80)
    print('CATEGORY BREAKDOWN BY METHOD AND REPLICATE AT 1% PRUNING')
    print('='*80)

    for method_name in ['Behler', 'Bispectrum', 'SOAP']:
        print(f'\n{method_name}:')
        print('-' * 80)
        print(f"{'Replicate':<12} {'Strained':<12} {'High Temp':<12} {'Normal':<12} {'Vacancy':<12} {'Surface':<12} {'Total':<8}")
        print('-' * 80)

        for rep_data in all_data[method_name]:
            rep = rep_data['replicate']
            counts = rep_data['counts']
            total = sum(counts.values())

            print(f"{rep:<12} "
                  f"{counts.get('Strained', 0):<12} "
                  f"{counts.get('High Temp (3374K)', 0):<12} "
                  f"{counts.get('Normal (300K)', 0):<12} "
                  f"{counts.get('Vacancy (300K)', 0):<12} "
                  f"{counts.get('Surface', 0):<12} "
                  f"{total:<8}")

    # Create visualization
    fig = plt.figure(figsize=(18, 12))

    # For each method, create a histogram showing category distribution across replicates
    method_names = ['Behler', 'Bispectrum', 'SOAP']

    for idx, method_name in enumerate(method_names):
        # Stacked bar chart for this method
        ax = plt.subplot(3, 3, idx*3 + 1)

        replicate_nums = [d['replicate'] for d in all_data[method_name]]

        # Prepare stacked data
        bottoms = np.zeros(len(replicate_nums))

        for cat in categories:
            counts = [d['counts'].get(cat, 0) for d in all_data[method_name]]
            ax.bar(range(len(replicate_nums)), counts, bottom=bottoms,
                   label=cat, color=colors[cat], edgecolor='white', linewidth=1)
            bottoms += np.array(counts)

        ax.set_ylabel('Number of Structures', fontsize=10, fontweight='bold')
        ax.set_xlabel('Replicate', fontsize=10, fontweight='bold')
        ax.set_title(f'{method_name} - Category Distribution per Replicate',
                     fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks(range(len(replicate_nums)))
        ax.set_xticklabels(replicate_nums, fontsize=8)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 2.5])

    # For each method, show histogram of category frequencies
    for idx, method_name in enumerate(method_names):
        ax = plt.subplot(3, 3, idx*3 + 2)

        # Count how many times each category appears across all replicates
        category_frequency = defaultdict(int)
        for rep_data in all_data[method_name]:
            for cat, count in rep_data['counts'].items():
                if count > 0:
                    category_frequency[cat] += count

        cats = [cat for cat in categories if category_frequency.get(cat, 0) > 0]
        freqs = [category_frequency[cat] for cat in cats]
        cat_colors = [colors[cat] for cat in cats]

        bars = ax.bar(range(len(cats)), freqs, color=cat_colors,
                      edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Total Selections Across Replicates', fontsize=10, fontweight='bold')
        ax.set_title(f'{method_name} - Total Category Selections',
                     fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c.replace(' (3374K)', '\n(3374K)').replace(' (300K)', '\n(300K)')
                            for c in cats], fontsize=8, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bar, freq in zip(bars, freqs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(freq)}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # For each method, show how consistent each category is across replicates
    for idx, method_name in enumerate(method_names):
        ax = plt.subplot(3, 3, idx*3 + 3)

        # For each category, count in how many replicates it appears
        category_replicate_count = defaultdict(int)
        for rep_data in all_data[method_name]:
            for cat, count in rep_data['counts'].items():
                if count > 0:
                    category_replicate_count[cat] += 1

        cats = [cat for cat in categories if category_replicate_count.get(cat, 0) > 0]
        counts = [category_replicate_count[cat] for cat in cats]
        cat_colors = [colors[cat] for cat in cats]

        bars = ax.barh(range(len(cats)), counts, color=cat_colors,
                       edgecolor='black', linewidth=1.5)
        ax.set_xlabel('# Replicates Selecting This Category', fontsize=10, fontweight='bold')
        ax.set_title(f'{method_name} - Category Consistency',
                     fontsize=11, fontweight='bold', pad=10)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=9, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim([0, 11])
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax.text(width + 0.2, i, f'{int(count)}/10',
                   va='center', fontsize=9, fontweight='bold')

    plt.suptitle('Structure Category Analysis at 1% Pruning - Per Replicate Breakdown\n' +
                 'Each replicate selects ~2 structures (1% of 214)',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    # Save
    output_path = '/Users/blaubach/claude_stuff/replicate_category_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\n\nVisualization saved to: {output_path}')

    # Summary statistics
    print('\n' + '='*80)
    print('SUMMARY STATISTICS')
    print('='*80)

    for method_name in ['Behler', 'Bispectrum', 'SOAP']:
        print(f'\n{method_name}:')

        # Count category appearances across replicates
        category_replicate_count = defaultdict(int)
        category_total_count = defaultdict(int)

        for rep_data in all_data[method_name]:
            for cat, count in rep_data['counts'].items():
                if count > 0:
                    category_replicate_count[cat] += 1
                    category_total_count[cat] += count

        for cat in categories:
            if cat in category_replicate_count:
                rep_count = category_replicate_count[cat]
                total_count = category_total_count[cat]
                avg_per_rep = total_count / 10  # 10 replicates
                print(f'  {cat:<20}: Selected in {rep_count}/10 replicates, '
                      f'{total_count} total selections, '
                      f'{avg_per_rep:.1f} avg per replicate')


if __name__ == '__main__':
    main()
