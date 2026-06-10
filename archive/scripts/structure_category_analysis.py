#!/usr/bin/env python3
"""
Analyze structure categories selected at 1% pruning across the three methods.
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


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


def get_structures_by_method():
    """Get structures for each method at 1%."""
    results = {}

    for method in ['behler', 'bispectrum', 'pruning']:
        method_name = 'SOAP' if method == 'pruning' else method.capitalize()
        files = glob.glob(f'{method}/replicates_structure_pruning_modified/si_structures_mean_std_1percent_replicate*.json')
        files = [f for f in files if not f.endswith('_info.json')]

        descriptions = set()
        for file in files:
            data = json.load(open(file))
            for entry in data:
                descriptions.add(entry['description'])

        results[method_name] = descriptions

    return results


def categorize_set(descriptions):
    """Categorize a set of descriptions."""
    categories = {}
    for desc in descriptions:
        cat = categorize_structure(desc)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(desc)
    return categories


def main():
    # Get data
    method_structures = get_structures_by_method()

    behler = method_structures['Behler']
    bispectrum = method_structures['Bispectrum']
    soap = method_structures['SOAP']

    # Calculate overlaps
    all_three = behler & bispectrum & soap
    union = behler | bispectrum | soap

    # Categorize
    union_cats = categorize_set(union)
    overlap_cats = categorize_set(all_three)
    behler_cats = categorize_set(behler)
    bispectrum_cats = categorize_set(bispectrum)
    soap_cats = categorize_set(soap)

    # Define consistent category order and colors
    cat_order = ['Strained', 'High Temp (3374K)', 'Normal (300K)', 'Vacancy (300K)', 'Surface']
    colors = {
        'Strained': '#E63946',
        'High Temp (3374K)': '#F77F00',
        'Normal (300K)': '#06A77D',
        'Vacancy (300K)': '#457B9D',
        'Surface': '#A23B72'
    }

    # Create figure
    fig = plt.figure(figsize=(18, 10))

    # 1. Pie chart - Union composition
    ax1 = plt.subplot(2, 3, 1)
    union_counts = [len(union_cats.get(cat, [])) for cat in cat_order]
    union_colors = [colors[cat] for cat in cat_order]

    wedges, texts, autotexts = ax1.pie(union_counts, labels=cat_order, autopct='%1.1f%%',
                                         colors=union_colors, startangle=90,
                                         textprops={'fontsize': 10, 'weight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')

    ax1.set_title('All Structures Selected at 1%\n(Union of 3 methods, n=16)',
                  fontsize=12, fontweight='bold', pad=15)

    # 2. Pie chart - Core overlap composition
    ax2 = plt.subplot(2, 3, 2)
    overlap_counts = [len(overlap_cats.get(cat, [])) for cat in cat_order]
    overlap_colors = [colors[cat] for cat in cat_order]

    wedges, texts, autotexts = ax2.pie(overlap_counts, labels=cat_order, autopct='%1.1f%%',
                                         colors=overlap_colors, startangle=90,
                                         textprops={'fontsize': 10, 'weight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')

    ax2.set_title('Core Agreement\n(All 3 methods agree, n=10)',
                  fontsize=12, fontweight='bold', pad=15)

    # 3. Stacked bar showing category breakdown by method
    ax3 = plt.subplot(2, 3, 3)

    methods = ['Behler', 'Bispectrum', 'SOAP']
    method_data = [behler_cats, bispectrum_cats, soap_cats]

    x_pos = np.arange(len(methods))
    width = 0.6
    bottoms = np.zeros(len(methods))

    for cat in cat_order:
        counts = [len(md.get(cat, [])) for md in method_data]
        ax3.bar(x_pos, counts, width, label=cat, bottom=bottoms, color=colors[cat],
                edgecolor='white', linewidth=1.5)
        bottoms += counts

    ax3.set_ylabel('Number of Structures', fontsize=11, fontweight='bold')
    ax3.set_title('Category Breakdown by Method', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(axis='y', alpha=0.2, linestyle='--')

    # 4. Heatmap-style visualization showing agreement
    ax4 = plt.subplot(2, 3, 4)

    # Calculate for each category how many methods selected structures from it
    categories_list = cat_order
    heatmap_data = []

    for cat in categories_list:
        in_behler = 1 if cat in behler_cats and len(behler_cats[cat]) > 0 else 0
        in_bispec = 1 if cat in bispectrum_cats and len(bispectrum_cats[cat]) > 0 else 0
        in_soap = 1 if cat in soap_cats and len(soap_cats[cat]) > 0 else 0
        total = in_behler + in_bispec + in_soap

        behler_count = len(behler_cats.get(cat, []))
        bispec_count = len(bispectrum_cats.get(cat, []))
        soap_count = len(soap_cats.get(cat, []))

        heatmap_data.append([behler_count, bispec_count, soap_count])

    heatmap_array = np.array(heatmap_data)
    im = ax4.imshow(heatmap_array, cmap='YlOrRd', aspect='auto')

    ax4.set_xticks(np.arange(3))
    ax4.set_yticks(np.arange(len(categories_list)))
    ax4.set_xticklabels(['Behler', 'Bispectrum', 'SOAP'], fontweight='bold')
    ax4.set_yticklabels(categories_list, fontweight='bold')

    # Add text annotations
    for i in range(len(categories_list)):
        for j in range(3):
            text = ax4.text(j, i, int(heatmap_array[i, j]),
                           ha="center", va="center", color="white", fontsize=12, fontweight='bold')

    ax4.set_title('Structures per Category by Method', fontsize=12, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax4, label='Count')

    # 5. Agreement ratio - what fraction of each category is in the overlap
    ax5 = plt.subplot(2, 3, 5)

    agreement_ratios = []
    for cat in cat_order:
        union_count = len(union_cats.get(cat, []))
        overlap_count = len(overlap_cats.get(cat, []))
        ratio = 100 * overlap_count / union_count if union_count > 0 else 0
        agreement_ratios.append(ratio)

    bars = ax5.barh(cat_order, agreement_ratios, color=[colors[cat] for cat in cat_order],
                     edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars, agreement_ratios)):
        union_count = len(union_cats.get(cat_order[i], []))
        overlap_count = len(overlap_cats.get(cat_order[i], []))
        ax5.text(ratio + 2, i, f'{overlap_count}/{union_count}',
                va='center', fontsize=10, fontweight='bold')

    ax5.set_xlabel('Agreement Rate (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Category Agreement Across Methods\n(% in core overlap)',
                  fontsize=12, fontweight='bold', pad=10)
    ax5.set_xlim([0, 110])
    ax5.grid(axis='x', alpha=0.2, linestyle='--')
    ax5.invert_yaxis()

    # 6. Detailed breakdown table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    table_data = [['Category', 'Union', 'Core\nOverlap', 'Agreement\nRate', 'Dominant\nMethod']]

    for cat in cat_order:
        union_count = len(union_cats.get(cat, []))
        overlap_count = len(overlap_cats.get(cat, []))
        agreement = f'{100*overlap_count/union_count:.0f}%' if union_count > 0 else 'N/A'

        # Find which method has most of this category
        behler_count = len(behler_cats.get(cat, []))
        bispec_count = len(bispectrum_cats.get(cat, []))
        soap_count = len(soap_cats.get(cat, []))

        max_count = max(behler_count, bispec_count, soap_count)
        dominant = []
        if behler_count == max_count and behler_count > 0:
            dominant.append('B')
        if bispec_count == max_count and bispec_count > 0:
            dominant.append('Bi')
        if soap_count == max_count and soap_count > 0:
            dominant.append('S')
        dominant_str = ','.join(dominant)

        table_data.append([
            cat,
            union_count,
            overlap_count,
            agreement,
            dominant_str
        ])

    table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Color rows by category
    for i in range(1, len(table_data)):
        cat = table_data[i][0]
        for j in range(5):
            if j == 0:  # Category column
                table[(i, j)].set_facecolor(colors.get(cat, '#FFFFFF'))
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')

    # Style header row
    for j in range(5):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(weight='bold', color='white')

    ax6.set_title('Summary Statistics by Category', fontsize=12, fontweight='bold', pad=20)

    # Overall title
    plt.suptitle('Structure Category Analysis at 1% Pruning\nComparison of Behler, Bispectrum, and SOAP',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = '/Users/blaubach/claude_stuff/structure_category_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved to: {output_path}')

    # Print detailed breakdown
    print('\n' + '='*80)
    print('DETAILED CATEGORY BREAKDOWN AT 1% PRUNING')
    print('='*80)

    print('\nALL STRUCTURES (UNION):')
    for cat in cat_order:
        structures = union_cats.get(cat, [])
        print(f'\n{cat}: {len(structures)} structures')
        for s in sorted(structures):
            print(f'  - {s}')

    print('\n' + '='*80)
    print('CORE OVERLAP (ALL THREE METHODS AGREE):')
    for cat in cat_order:
        structures = overlap_cats.get(cat, [])
        if structures:
            print(f'\n{cat}: {len(structures)} structures')
            for s in sorted(structures):
                print(f'  - {s}')


if __name__ == '__main__':
    main()
