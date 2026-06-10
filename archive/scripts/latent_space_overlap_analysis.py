#!/usr/bin/env python3
"""
Compare overlap between behler, bispectrum, and SOAP (pruning) latent spaces
across pruning percentages 1-10% with replicates.
"""

import json
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def get_structure_ids(json_path):
    """Extract unique structure identifiers from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create unique identifier using structure info
    structure_ids = set()
    for entry in data:
        # Use description and group as identifier
        # Could also use structure properties if needed
        struct_id = (entry.get('description', ''),
                     entry.get('group', ''),
                     entry.get('num_atoms', ''))
        structure_ids.add(struct_id)

    return structure_ids


def collect_structures_for_percent(percent, base_dirs):
    """
    Collect all structures across all replicates for a given pruning percentage.

    Args:
        percent: Pruning percentage (1-10)
        base_dirs: Dict mapping method names to their base directories

    Returns:
        Dict mapping method names to sets of structure IDs
    """
    method_structures = {}

    for method, base_dir in base_dirs.items():
        all_structures = set()

        # Find all replicate files for this percentage (exclude _info.json files)
        pattern = os.path.join(
            base_dir,
            f'replicates_structure_pruning_modified/si_structures_mean_std_{percent}percent_replicate*.json'
        )
        all_files = glob.glob(pattern)
        files = [f for f in all_files if not f.endswith('_info.json')]

        # Combine structures from all replicates
        for json_file in files:
            structures = get_structure_ids(json_file)
            all_structures.update(structures)

        method_structures[method] = all_structures
        print(f"{method} - {percent}%: {len(all_structures)} unique structures from {len(files)} replicates")

    return method_structures


def calculate_overlap_stats(sets_dict):
    """Calculate overlap statistics between three sets."""
    behler = sets_dict['Behler']
    bispectrum = sets_dict['Bispectrum']
    soap = sets_dict['SOAP']

    # Three-way overlap
    all_three = behler & bispectrum & soap

    # Two-way overlaps (exclusive)
    behler_bispectrum_only = (behler & bispectrum) - soap
    behler_soap_only = (behler & soap) - bispectrum
    bispectrum_soap_only = (bispectrum & soap) - behler

    # Unique to each
    behler_only = behler - bispectrum - soap
    bispectrum_only = bispectrum - behler - soap
    soap_only = soap - behler - bispectrum

    total_union = behler | bispectrum | soap

    stats = {
        'all_three': len(all_three),
        'behler_bispectrum_only': len(behler_bispectrum_only),
        'behler_soap_only': len(behler_soap_only),
        'bispectrum_soap_only': len(bispectrum_soap_only),
        'behler_only': len(behler_only),
        'bispectrum_only': len(bispectrum_only),
        'soap_only': len(soap_only),
        'total_union': len(total_union),
        'behler_total': len(behler),
        'bispectrum_total': len(bispectrum),
        'soap_total': len(soap)
    }

    return stats


def main():
    base_dirs = {
        'Behler': '/Users/blaubach/claude_stuff/behler',
        'Bispectrum': '/Users/blaubach/claude_stuff/bispectrum',
        'SOAP': '/Users/blaubach/claude_stuff/pruning'
    }

    # Verify directories exist
    for method, path in base_dirs.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")

    # Collect data for each percentage
    percentages = range(1, 11)
    all_stats = {}

    print("Collecting structures for each pruning percentage...\n")
    for percent in percentages:
        print(f"Processing {percent}%:")
        method_structures = collect_structures_for_percent(percent, base_dirs)
        stats = calculate_overlap_stats(method_structures)
        all_stats[percent] = stats
        print(f"  Total union: {stats['total_union']}, All three: {stats['all_three']}\n")

    # Create visualization
    fig = plt.figure(figsize=(18, 10))

    x = list(percentages)

    # Prepare data
    all_three = [all_stats[p]['all_three'] for p in percentages]
    behler_bispectrum = [all_stats[p]['behler_bispectrum_only'] for p in percentages]
    behler_soap = [all_stats[p]['behler_soap_only'] for p in percentages]
    bispectrum_soap = [all_stats[p]['bispectrum_soap_only'] for p in percentages]
    behler_only = [all_stats[p]['behler_only'] for p in percentages]
    bispectrum_only = [all_stats[p]['bispectrum_only'] for p in percentages]
    soap_only = [all_stats[p]['soap_only'] for p in percentages]
    total_union = [all_stats[p]['total_union'] for p in percentages]
    behler_totals = [all_stats[p]['behler_total'] for p in percentages]
    bispectrum_totals = [all_stats[p]['bispectrum_total'] for p in percentages]
    soap_totals = [all_stats[p]['soap_total'] for p in percentages]

    # Color scheme
    color_all_three = '#2E86AB'  # Blue - shared by all
    color_behler_bispec = '#A23B72'  # Purple - Behler + Bispectrum
    color_behler_soap = '#F18F01'  # Orange - Behler + SOAP
    color_bispec_soap = '#C73E1D'  # Red - Bispectrum + SOAP
    color_behler = '#6A994E'  # Green - Behler only
    color_bispec = '#BC4B51'  # Dark red - Bispectrum only
    color_soap = '#8B5A3C'  # Brown - SOAP only

    # 1. 100% Stacked bar showing proportions
    ax1 = plt.subplot(2, 3, 1)

    # Convert to percentages
    all_three_pct = [100 * all_three[i] / total_union[i] for i in range(len(percentages))]
    behler_bispectrum_pct = [100 * behler_bispectrum[i] / total_union[i] for i in range(len(percentages))]
    behler_soap_pct = [100 * behler_soap[i] / total_union[i] for i in range(len(percentages))]
    bispectrum_soap_pct = [100 * bispectrum_soap[i] / total_union[i] for i in range(len(percentages))]
    behler_only_pct = [100 * behler_only[i] / total_union[i] for i in range(len(percentages))]
    bispectrum_only_pct = [100 * bispectrum_only[i] / total_union[i] for i in range(len(percentages))]
    soap_only_pct = [100 * soap_only[i] / total_union[i] for i in range(len(percentages))]

    ax1.bar(x, all_three_pct, label='All Three Methods', color=color_all_three, edgecolor='white', linewidth=0.5)

    bottom1 = np.array(all_three_pct)
    ax1.bar(x, behler_bispectrum_pct, bottom=bottom1, label='Behler ∩ Bispectrum',
            color=color_behler_bispec, edgecolor='white', linewidth=0.5)

    bottom2 = bottom1 + np.array(behler_bispectrum_pct)
    ax1.bar(x, behler_soap_pct, bottom=bottom2, label='Behler ∩ SOAP',
            color=color_behler_soap, edgecolor='white', linewidth=0.5)

    bottom3 = bottom2 + np.array(behler_soap_pct)
    ax1.bar(x, bispectrum_soap_pct, bottom=bottom3, label='Bispectrum ∩ SOAP',
            color=color_bispec_soap, edgecolor='white', linewidth=0.5)

    bottom4 = bottom3 + np.array(bispectrum_soap_pct)
    ax1.bar(x, behler_only_pct, bottom=bottom4, label='Behler Only',
            color=color_behler, edgecolor='white', linewidth=0.5)

    bottom5 = bottom4 + np.array(behler_only_pct)
    ax1.bar(x, bispectrum_only_pct, bottom=bottom5, label='Bispectrum Only',
            color=color_bispec, edgecolor='white', linewidth=0.5)

    bottom6 = bottom5 + np.array(bispectrum_only_pct)
    ax1.bar(x, soap_only_pct, bottom=bottom6, label='SOAP Only',
            color=color_soap, edgecolor='white', linewidth=0.5)

    ax1.set_xlabel('Pruning Percentage (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Proportion of Total Union (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Overlap Composition (Normalized)', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax1.set_xticks(x)
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.2, linestyle='--')

    # 2. Grouped bar chart showing unique vs shared
    ax2 = plt.subplot(2, 3, 2)

    unique_counts = [behler_only[i] + bispectrum_only[i] + soap_only[i] for i in range(len(percentages))]
    shared_two = [behler_bispectrum[i] + behler_soap[i] + bispectrum_soap[i] for i in range(len(percentages))]

    width = 0.25
    x_pos = np.arange(len(x))

    ax2.bar(x_pos - width, unique_counts, width, label='Unique to One Method', color='#E63946', alpha=0.8)
    ax2.bar(x_pos, shared_two, width, label='Shared by Two Methods', color='#F77F00', alpha=0.8)
    ax2.bar(x_pos + width, all_three, width, label='Shared by All Three', color='#06A77D', alpha=0.8)

    ax2.set_xlabel('Pruning Percentage (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Structures', fontsize=11, fontweight='bold')
    ax2.set_title('Unique vs Shared Structures', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.2, linestyle='--')

    # 3. Method diversity - unique structures per method
    ax3 = plt.subplot(2, 3, 3)

    ax3.plot(x, behler_totals, marker='o', label='Behler', linewidth=2.5, markersize=8, color='#6A994E')
    ax3.plot(x, bispectrum_totals, marker='s', label='Bispectrum', linewidth=2.5, markersize=8, color='#BC4B51')
    ax3.plot(x, soap_totals, marker='^', label='SOAP', linewidth=2.5, markersize=8, color='#8B5A3C')

    ax3.fill_between(x, behler_totals, alpha=0.1, color='#6A994E')
    ax3.fill_between(x, bispectrum_totals, alpha=0.1, color='#BC4B51')
    ax3.fill_between(x, soap_totals, alpha=0.1, color='#8B5A3C')

    ax3.set_xlabel('Pruning Percentage (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Unique Structures Across Replicates', fontsize=11, fontweight='bold')
    ax3.set_title('Method Diversity', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(fontsize=10, loc='upper left')
    ax3.set_xticks(x)
    ax3.grid(True, alpha=0.2, linestyle='--')

    # 4. Overlap percentage trends
    ax4 = plt.subplot(2, 3, 4)

    overlap_pct = [100 * all_three[i] / total_union[i] for i in range(len(percentages))]

    ax4.plot(x, overlap_pct, marker='o', linewidth=3, markersize=10, color='#2E86AB',
             markerfacecolor='#2E86AB', markeredgecolor='white', markeredgewidth=2)
    ax4.fill_between(x, overlap_pct, alpha=0.3, color='#2E86AB')

    ax4.set_xlabel('Pruning Percentage (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Three-Way Overlap (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Core Agreement Between All Methods', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.grid(True, alpha=0.2, linestyle='--')
    ax4.set_ylim([0, max(overlap_pct) * 1.2])

    # Add value labels
    for i, (xi, yi) in enumerate(zip(x, overlap_pct)):
        ax4.text(xi, yi + 2, f'{yi:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 5. Pairwise overlap comparison
    ax5 = plt.subplot(2, 3, 5)

    behler_bispec_pct = [100 * (all_three[i] + behler_bispectrum[i]) / total_union[i] for i in range(len(percentages))]
    behler_soap_pct = [100 * (all_three[i] + behler_soap[i]) / total_union[i] for i in range(len(percentages))]
    bispec_soap_pct = [100 * (all_three[i] + bispectrum_soap[i]) / total_union[i] for i in range(len(percentages))]

    ax5.plot(x, behler_bispec_pct, marker='o', label='Behler ∩ Bispectrum', linewidth=2.5, markersize=8)
    ax5.plot(x, behler_soap_pct, marker='s', label='Behler ∩ SOAP', linewidth=2.5, markersize=8)
    ax5.plot(x, bispec_soap_pct, marker='^', label='Bispectrum ∩ SOAP', linewidth=2.5, markersize=8)

    ax5.set_xlabel('Pruning Percentage (%)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Pairwise Overlap (% of Union)', fontsize=11, fontweight='bold')
    ax5.set_title('Pairwise Method Agreement', fontsize=12, fontweight='bold', pad=10)
    ax5.legend(fontsize=9)
    ax5.set_xticks(x)
    ax5.grid(True, alpha=0.2, linestyle='--')
    ax5.set_ylim([0, 100])

    # 6. Absolute stacked bar
    ax6 = plt.subplot(2, 3, 6)

    ax6.bar(x, all_three, label='All Three Methods', color=color_all_three, edgecolor='white', linewidth=0.5)

    bottom1 = np.array(all_three)
    ax6.bar(x, behler_bispectrum, bottom=bottom1, label='Behler ∩ Bispectrum',
            color=color_behler_bispec, edgecolor='white', linewidth=0.5)

    bottom2 = bottom1 + np.array(behler_bispectrum)
    ax6.bar(x, behler_soap, bottom=bottom2, label='Behler ∩ SOAP',
            color=color_behler_soap, edgecolor='white', linewidth=0.5)

    bottom3 = bottom2 + np.array(behler_soap)
    ax6.bar(x, bispectrum_soap, bottom=bottom3, label='Bispectrum ∩ SOAP',
            color=color_bispec_soap, edgecolor='white', linewidth=0.5)

    bottom4 = bottom3 + np.array(bispectrum_soap)
    ax6.bar(x, behler_only, bottom=bottom4, label='Behler Only',
            color=color_behler, edgecolor='white', linewidth=0.5)

    bottom5 = bottom4 + np.array(behler_only)
    ax6.bar(x, bispectrum_only, bottom=bottom5, label='Bispectrum Only',
            color=color_bispec, edgecolor='white', linewidth=0.5)

    bottom6 = bottom5 + np.array(bispectrum_only)
    ax6.bar(x, soap_only, bottom=bottom6, label='SOAP Only',
            color=color_soap, edgecolor='white', linewidth=0.5)

    ax6.set_xlabel('Pruning Percentage (%)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Number of Structures', fontsize=11, fontweight='bold')
    ax6.set_title('Overlap Composition (Absolute Counts)', fontsize=12, fontweight='bold', pad=10)
    ax6.legend(fontsize=8, loc='upper left')
    ax6.set_xticks(x)
    ax6.grid(axis='y', alpha=0.2, linestyle='--')

    plt.suptitle('Latent Space Overlap Analysis: Behler, Bispectrum, and SOAP\n(Cumulative across all replicates, 1-10% pruning)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path = '/Users/blaubach/claude_stuff/latent_space_overlap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    # Print detailed statistics
    print("\n" + "="*80)
    print("DETAILED OVERLAP STATISTICS")
    print("="*80)
    for percent in percentages:
        stats = all_stats[percent]
        print(f"\n{percent}% Pruning:")
        print(f"  Total union: {stats['total_union']}")
        print(f"  Behler total: {stats['behler_total']}")
        print(f"  Bispectrum total: {stats['bispectrum_total']}")
        print(f"  SOAP total: {stats['soap_total']}")
        print(f"  All three: {stats['all_three']} ({100*stats['all_three']/stats['total_union']:.1f}%)")
        print(f"  Behler + Bispectrum only: {stats['behler_bispectrum_only']}")
        print(f"  Behler + SOAP only: {stats['behler_soap_only']}")
        print(f"  Bispectrum + SOAP only: {stats['bispectrum_soap_only']}")
        print(f"  Behler only: {stats['behler_only']}")
        print(f"  Bispectrum only: {stats['bispectrum_only']}")
        print(f"  SOAP only: {stats['soap_only']}")


if __name__ == '__main__':
    main()
