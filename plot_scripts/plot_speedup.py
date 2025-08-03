import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Load the experiment results
exp1_df = pd.read_csv('results/experiment1_results.csv')
exp2_df = pd.read_csv('results/experiment2_results.csv')

# Create a dictionary to store Experiment 1 results for easy lookup
exp1_times = dict(zip(exp1_df['seq_len'], exp1_df['time_ms']))

# Calculate speedup for each configuration in Experiment 2
speedup_data = []
for _, row in exp2_df.iterrows():
    seq_len = row['seq_len']
    middle_dim = row['middle_dim']
    exp2_time = row['time_ms']
    
    # Get corresponding time from Experiment 1
    exp1_time = exp1_times[seq_len]
    
    # Calculate speedup: Experiment 1 time / Experiment 2 time
    # (higher values mean Experiment 2 is faster)
    speedup = exp1_time / exp2_time
    
    speedup_data.append((seq_len, middle_dim, speedup))

# Convert to DataFrame
speedup_df = pd.DataFrame(speedup_data, columns=['seq_len', 'middle_dim', 'speedup'])

# Save speedup data to CSV
speedup_df.to_csv('results/speedup_results.csv', index=False)

# Create heatmap of speedup
plt.figure(figsize=(12, 8))
pivot_df = speedup_df.pivot(index='seq_len', columns='middle_dim', values='speedup')

# Use a custom colormap to make speedup > 1 (improvement) green and speedup < 1 (slowdown) red
heatmap = plt.pcolormesh(
    np.array([int(x) for x in pivot_df.columns]),
    np.array([int(x) for x in pivot_df.index]),
    pivot_df.values,
    cmap='RdYlGn',  # Red (slow) to Yellow (neutral) to Green (fast)
    vmin=0.5,       # Adjust as needed for your data
    vmax=5.0        # Adjust as needed for your data
)
plt.colorbar(heatmap, label='Speedup Factor (higher is better)')
plt.xlabel('Middle Dimension', fontsize=12)
plt.ylabel('Sequence Length', fontsize=12)
plt.title('Speedup of SVD', fontsize=14)
plt.xscale('log', base=2)
plt.yscale('log', base=2)

# Get the list of middle dimensions and sequence lengths from the data
middle_dims = sorted(speedup_df['middle_dim'].unique().astype(int))
seq_lengths = sorted(speedup_df['seq_len'].unique().astype(int))

plt.xticks(middle_dims, [str(x) for x in middle_dims], rotation=45)
plt.yticks(seq_lengths, [str(x) for x in seq_lengths])
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('results/speedup_heatmap.png', dpi=300)

# Line plot showing speedup vs middle dimension for different sequence lengths
plt.figure(figsize=(12, 8))
for seq_len in seq_lengths:
    subset = speedup_df[speedup_df['seq_len'] == seq_len]
    plt.plot(subset['middle_dim'], subset['speedup'], 'o-', linewidth=2, label=f'seq_len={seq_len}')

plt.xscale('log', base=2)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel('Middle Dimension', fontsize=12)
plt.ylabel('Speedup Factor (No SVD time / SVD time)', fontsize=12)
plt.title('Speedup of SVD', fontsize=14)
plt.legend(fontsize=10)
plt.xticks(middle_dims, [str(x) for x in middle_dims], rotation=45)

# Add a horizontal line at y=1 to indicate the baseline (no speedup)
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline (no speedup)')

plt.tight_layout()
plt.savefig('results/speedup_lines.png', dpi=300)

# Bar chart for optimal middle dimension for each sequence length
optimal_speedups = []
for seq_len in seq_lengths:
    subset = speedup_df[speedup_df['seq_len'] == seq_len]
    max_speedup_row = subset.loc[subset['speedup'].idxmax()]
    optimal_speedups.append((
        seq_len, 
        max_speedup_row['middle_dim'],
        max_speedup_row['speedup']
    ))

opt_df = pd.DataFrame(optimal_speedups, columns=['seq_len', 'best_middle_dim', 'max_speedup'])
opt_df.to_csv('results/optimal_speedup.csv', index=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(seq_lengths)), opt_df['max_speedup'], color='teal')

# Add middle dimension labels to each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2., 
        height + 0.1,
        f'mid_dim={int(opt_df.iloc[i]["best_middle_dim"])}',
        ha='center', 
        va='bottom', 
        rotation=0,
        fontsize=10
    )

plt.xticks(range(len(seq_lengths)), [str(x) for x in seq_lengths])
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('Maximum Speedup Factor', fontsize=12)
plt.title('Optimal SVD Speedup and Middle Dimension for Each Sequence Length', fontsize=14)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/optimal_speedup_bar.png', dpi=300)

print("Speedup analysis complete. Results saved to the 'results' directory.") 