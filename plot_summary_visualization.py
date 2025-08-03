import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Load the experiment results
exp1_df = pd.read_csv('results/experiment1_results.csv')
exp2_df = pd.read_csv('results/experiment2_results.csv')
speedup_df = pd.read_csv('results/speedup_results.csv')
opt_df = pd.read_csv('results/optimal_speedup.csv')

# Create a figure with a grid layout for multiple plots
plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=plt.gcf())

# Get the sequence lengths and middle dimensions
seq_lengths = sorted(speedup_df['seq_len'].unique())
middle_dims = sorted(speedup_df['middle_dim'].unique())

# 1. Top-Left: Original execution times from both experiments
ax1 = plt.subplot(gs[0, 0])
plt.plot(exp1_df['seq_len'], exp1_df['time_ms'], 'o-', linewidth=2, 
         color='blue', label='No SVD (Baseline)')

# Add lines for each middle dimension in Experiment 2
for mid_dim in [64, 256, 1024, 4096]:  # Select representative middle dimensions
    subset = exp2_df[exp2_df['middle_dim'] == mid_dim]
    plt.plot(subset['seq_len'], subset['time_ms'], 'o--', linewidth=1.5, 
             label=f'SVD (mid_dim={mid_dim})')

plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.title('Matrix Multiplication Execution Times', fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.xticks(seq_lengths, [str(x) for x in seq_lengths], rotation=45)

# 2. Top-Right: Speedup across sequence lengths for different middle dimensions
ax2 = plt.subplot(gs[0, 1])
# Plot the optimal speedup
plt.plot(opt_df['seq_len'], opt_df['max_speedup'], 'o-', linewidth=3, 
         color='green', label='Optimal Middle Dimension')

# Add lines for specific middle dimensions
for mid_dim in [64, 256, 1024, 4096]:  # Select representative middle dimensions
    subset = speedup_df[speedup_df['middle_dim'] == mid_dim]
    plt.plot(subset['seq_len'], subset['speedup'], '--', linewidth=1.5, 
             label=f'mid_dim={mid_dim}')

plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Baseline (no speedup)')
plt.xscale('log', base=2)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('Speedup Factor', fontsize=12)
plt.title('SVD Speedup vs. Sequence Length', fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.xticks(seq_lengths, [str(x) for x in seq_lengths], rotation=45)

# 3. Bottom-Left: Heatmap of speedup
ax3 = plt.subplot(gs[1, 0])
pivot_df = speedup_df.pivot(index='seq_len', columns='middle_dim', values='speedup')

heatmap = plt.pcolormesh(
    np.array([int(x) for x in pivot_df.columns]),
    np.array([int(x) for x in pivot_df.index]),
    pivot_df.values,
    cmap='RdYlGn',
    vmin=0.5,
    vmax=5.0
)
plt.colorbar(heatmap, label='Speedup Factor')
plt.xlabel('Middle Dimension', fontsize=12)
plt.ylabel('Sequence Length', fontsize=12)
plt.title('SVD Speedup Heatmap', fontsize=14)
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xticks(middle_dims, [str(x) for x in middle_dims], rotation=45)
plt.yticks(seq_lengths, [str(x) for x in seq_lengths])

# 4. Bottom-Right: Best middle dimension and speedup for each sequence length
ax4 = plt.subplot(gs[1, 1])

# Create a twin axis for showing both best middle dimension and speedup
ax4_twin = ax4.twinx()

# Bars for speedup
bars = ax4.bar(np.arange(len(seq_lengths)), opt_df['max_speedup'], color='teal', alpha=0.7, label='Max Speedup')
ax4.set_ylabel('Maximum Speedup Factor', fontsize=12)
ax4.set_ylim(0, max(opt_df['max_speedup']) * 1.2)

# Line for best middle dimension
ax4_twin.plot(np.arange(len(seq_lengths)), opt_df['best_middle_dim'], 'o-', color='red', linewidth=2, label='Best Middle Dimension')
ax4_twin.set_ylabel('Best Middle Dimension', fontsize=12, color='red')
ax4_twin.set_ylim(0, max(opt_df['best_middle_dim']) * 1.2)
ax4_twin.tick_params(axis='y', colors='red')

# Add value labels above the bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}x', ha='center', va='bottom', fontsize=10)

# Add middle dimension labels to the line
for i, val in enumerate(opt_df['best_middle_dim']):
    ax4_twin.text(i, val + 10, f'{int(val)}', ha='center', va='bottom', 
                  color='red', fontsize=10, rotation=0)

# Setup x-ticks
ax4.set_xticks(np.arange(len(seq_lengths)))
ax4.set_xticklabels([str(int(x)) for x in seq_lengths], rotation=45)
ax4.set_xlabel('Sequence Length', fontsize=12)
ax4.set_title('Optimal Middle Dimension and Maximum Speedup', fontsize=14)

# Add a combined legend
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# Add annotations highlighting key insights
plt.figtext(0.5, 0.01, 
            "Key Insight: Smaller middle dimensions (64-256) provide optimal performance, especially for longer sequences. "
            "SVD speedup increases dramatically with sequence length, reaching 9.46x for seq_len=4096.",
            ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.5))

plt.suptitle('Matrix Multiplication Performance Analysis: SVD Factorization Speedup', fontsize=16, y=0.99)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig('results/summary_visualization.png', dpi=300, bbox_inches='tight')
print("Summary visualization saved to 'results/summary_visualization.png'") 