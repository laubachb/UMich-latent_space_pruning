#!/usr/bin/env python3
"""
Simplified structure category analysis at 1% pruning.
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np


def categorize_structure(description):
    """Categorize a structure based on its description."""
    if 'strain' in description.lower() or 'mode' in description.lower():
        return 'Strained'
    elif 'vacancy' in description.lower():
        if '3374 K' in description:
            return 'High Temp\n(3374K)'
        else:
            return 'Vacancy\n(300K)'
    elif 'surface' in description.lower():
        return 'Surface'
    elif '3374 K' in description:
        return 'High Temp\n(3374K)'
    elif '300 K' in description:
        return 'Normal\n(300K)'
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
    """Categorize a set of descriptions and return counts."""
    categories = {}
    for desc in descriptions:
        cat = categorize_structure(desc)
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
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

    # Define categories
    cat_order = ['Strained', 'High Temp\n(3374K)', 'Normal\n(300K)', 'Vacancy\n(300K)', 'Surface']
    colors = {
        'Strained': '#E63946',
        'High Temp\n(3374K)': '#F77F00',
        'Normal\n(300K)': '#06A77D',
        'Vacancy\n(300K)': '#457B9D',
        'Surface': '#A23B72'
    }

    # Categorize
    union_cats = categorize_set(union)
    overlap_cats = categorize_set(all_three)

    # Create simple visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Simple bar chart: What did we find?
    ax1 = axes[0, 0]
    union_counts = [union_cats.get(cat, 0) for cat in cat_order]
    overlap_counts = [overlap_cats.get(cat, 0) for cat in cat_order]

    x = np.arange(len(cat_order))
    width = 0.35

    bars1 = ax1.bar(x - width/2, union_counts, width, label='At least 1 method selected',
                    color=[colors[cat] for cat in cat_order], alpha=0.6, edgecolor='black', linewidth=2)
    bars2 = ax1.bar(x + width/2, overlap_counts, width, label='All 3 methods agreed',
                    color=[colors[cat] for cat in cat_order], alpha=1.0, edgecolor='black', linewidth=2)

    ax1.set_ylabel('Number of Structures', fontsize=12, fontweight='bold')
    ax1.set_title('What Types of Structures Were Selected at 1% Pruning?',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_order, fontsize=10, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, max(union_counts) + 1])

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Agreement percentage
    ax2 = axes[0, 1]
    agreement_pcts = []
    labels_with_counts = []

    for cat in cat_order:
        union_count = union_cats.get(cat, 0)
        overlap_count = overlap_cats.get(cat, 0)
        if union_count > 0:
            pct = 100 * overlap_count / union_count
            agreement_pcts.append(pct)
            labels_with_counts.append(f'{cat.replace(chr(10), " ")}\n({overlap_count}/{union_count})')
        else:
            agreement_pcts.append(0)
            labels_with_counts.append(f'{cat.replace(chr(10), " ")}\n(0/0)')

    bars = ax2.barh(range(len(cat_order)), agreement_pcts,
                     color=[colors[cat] for cat in cat_order],
                     edgecolor='black', linewidth=2)

    ax2.set_yticks(range(len(cat_order)))
    ax2.set_yticklabels(labels_with_counts, fontsize=9, fontweight='bold')
    ax2.set_xlabel('Agreement Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('How Much Do Methods Agree on Each Category?',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlim([0, 110])
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, agreement_pcts)):
        if pct > 0:
            ax2.text(pct + 2, i, f'{pct:.0f}%',
                    va='center', fontsize=11, fontweight='bold')

    # 3. Venn-style representation for each category
    ax3 = axes[1, 0]
    ax3.axis('off')

    # Create text summary
    summary_text = "DETAILED BREAKDOWN:\n\n"
    summary_text += f"Total unique structures selected: {len(union)}\n"
    summary_text += f"Structures all 3 methods agree on: {len(all_three)}\n"
    summary_text += f"Overall agreement rate: {100*len(all_three)/len(union):.1f}%\n\n"

    summary_text += "By Category:\n"
    for cat in cat_order:
        cat_name = cat.replace('\n', ' ')
        union_count = union_cats.get(cat, 0)
        overlap_count = overlap_cats.get(cat, 0)
        if union_count > 0:
            pct = 100 * overlap_count / union_count
            summary_text += f"\n{cat_name}:\n"
            summary_text += f"  • Total selected: {union_count}\n"
            summary_text += f"  • All 3 agreed: {overlap_count}\n"
            summary_text += f"  • Agreement: {pct:.0f}%\n"

    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 4. Key insights
    ax4 = axes[1, 1]
    ax4.axis('off')

    insights = "KEY INSIGHTS:\n\n"

    # Find category with highest agreement
    max_agreement_cat = None
    max_agreement_pct = 0
    for cat in cat_order:
        union_count = union_cats.get(cat, 0)
        overlap_count = overlap_cats.get(cat, 0)
        if union_count > 0:
            pct = 100 * overlap_count / union_count
            if pct > max_agreement_pct:
                max_agreement_pct = pct
                max_agreement_cat = cat.replace('\n', ' ')

    # Find most common category
    max_union_cat = max(cat_order, key=lambda c: union_cats.get(c, 0))
    max_union_count = union_cats.get(max_union_cat, 0)

    insights += f"1. STRAINED STRUCTURES DOMINATE:\n"
    insights += f"   • {union_cats.get('Strained', 0)} out of {len(union)} selected ({100*union_cats.get('Strained', 0)/len(union):.0f}%)\n"
    insights += f"   • These are the most common 'outliers'\n\n"

    insights += f"2. PERFECT AGREEMENT ON:\n"
    for cat in cat_order:
        union_count = union_cats.get(cat, 0)
        overlap_count = overlap_cats.get(cat, 0)
        if union_count > 0 and union_count == overlap_count:
            cat_name = cat.replace('\n', ' ')
            insights += f"   • {cat_name}: {overlap_count}/{union_count} (100%)\n"
    insights += f"   All methods see these as critical!\n\n"

    insights += f"3. MOST DISAGREEMENT ON:\n"
    min_agreement_cat = None
    min_agreement_pct = 100
    for cat in cat_order:
        union_count = union_cats.get(cat, 0)
        overlap_count = overlap_cats.get(cat, 0)
        if union_count > 0:
            pct = 100 * overlap_count / union_count
            if pct < min_agreement_pct:
                min_agreement_pct = pct
                min_agreement_cat = cat.replace('\n', ' ')

    if min_agreement_cat:
        insights += f"   • {min_agreement_cat}: {min_agreement_pct:.0f}% agreement\n"
        insights += f"   Different methods prioritize different\n"
        insights += f"   {min_agreement_cat.lower()} structures\n"

    ax4.text(0.05, 0.95, insights, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Understanding Structure Selection at 1% Pruning\nBehler vs Bispectrum vs SOAP',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = '/Users/blaubach/claude_stuff/simple_category_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Simplified figure saved to: {output_path}')


if __name__ == '__main__':
    main()
