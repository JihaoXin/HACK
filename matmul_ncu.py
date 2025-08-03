import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os

WARMUP_ITERATIONS = 5
NUM_RUNS = 3  # Number of times to run each experiment for averaging

# Define the ranges for experimentation
seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096]
middle_dims = [64, 128, 256, 512, 1024, 2048, 4096]
model_dim = 4096  # Keep model dimension fixed

# Data storage for results
results_exp1 = []  # Will store (seq_len, time_ms) for experiment 1
results_exp2 = []  # Will store (seq_len, middle_dim, time_ms) for experiment 2

def experiment1(seq_len):
    """
    Two consecutive matmuls, size [seq_len,model_dim]@[model_dim,model_dim]@[model_dim,seq_len]
    """
    # Initialize matrices on GPU
    A = torch.rand(seq_len, model_dim, dtype=torch.bfloat16, device='cuda')
    B = torch.rand(model_dim, model_dim, dtype=torch.bfloat16, device='cuda')
    C = torch.rand(model_dim, seq_len, dtype=torch.bfloat16, device='cuda')
    
    # warmup
    for _ in range(WARMUP_ITERATIONS):
        temp = A @ B
        result = temp @ C
    
    # The actual computation to be profiled
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        temp = A @ B
        result = temp @ C
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        times.append(elapsed_time)
    
    avg_time = sum(times) / len(times)
    print(f"Experiment 1 (seq_len={seq_len}): {avg_time:.3f} ms")
    return avg_time

def experiment2(seq_len, middle_dim):
    """
    Three matmuls: ([seq_len,model_dim]@[model_dim,middle_dim])@([middle_dim,model_dim]@[model_dim,seq_len])
    """
    # Initialize matrices on GPU
    A = torch.rand(seq_len, model_dim, dtype=torch.bfloat16, device='cuda')
    B = torch.rand(model_dim, middle_dim, dtype=torch.bfloat16, device='cuda')
    C = torch.rand(middle_dim, model_dim, dtype=torch.bfloat16, device='cuda')
    D = torch.rand(model_dim, seq_len, dtype=torch.bfloat16, device='cuda')
    
    # warmup
    for _ in range(WARMUP_ITERATIONS):
        left = A @ B
        right = C @ D
        result = left @ right
    
    # The actual computation to be profiled
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        left = A @ B
        right = C @ D
        result = left @ right
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        times.append(elapsed_time)
    
    avg_time = sum(times) / len(times)
    print(f"Experiment 2 (seq_len={seq_len}, middle_dim={middle_dim}): {avg_time:.3f} ms")
    return avg_time

def run_all_experiments():
    """Run all experiments with various sequence lengths and middle dimensions"""
    print("=== Running Experiment 1 with different sequence lengths ===")
    for seq_len in seq_lengths:
        time_ms = experiment1(seq_len)
        results_exp1.append((seq_len, time_ms))
    
    print("\n=== Running Experiment 2 with different sequence lengths and middle dimensions ===")
    for seq_len in seq_lengths:
        for mid_dim in middle_dims:
            time_ms = experiment2(seq_len, mid_dim)
            results_exp2.append((seq_len, mid_dim, time_ms))

def plot_results():
    """Create plots from the collected results"""
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save raw data to CSV
    exp1_df = pd.DataFrame(results_exp1, columns=['seq_len', 'time_ms'])
    exp1_df.to_csv('results/experiment1_results.csv', index=False)
    
    exp2_df = pd.DataFrame(results_exp2, columns=['seq_len', 'middle_dim', 'time_ms'])
    exp2_df.to_csv('results/experiment2_results.csv', index=False)
    
    # Plot Experiment 1 results
    plt.figure(figsize=(10, 6))
    plt.plot(exp1_df['seq_len'], exp1_df['time_ms'], 'o-', linewidth=2)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Experiment 1: Execution Time vs. Sequence Length', fontsize=14)
    plt.xticks(seq_lengths, [str(x) for x in seq_lengths], rotation=45)
    plt.tight_layout()
    plt.savefig('results/experiment1_plot.png', dpi=300)
    
    # Plot Experiment 2 results - Heatmap for seq_len vs middle_dim
    pivot_df = exp2_df.pivot(index='seq_len', columns='middle_dim', values='time_ms')
    
    plt.figure(figsize=(12, 8))
    heatmap = plt.pcolormesh(
        np.array([int(x) for x in pivot_df.columns]),
        np.array([int(x) for x in pivot_df.index]),
        pivot_df.values,
        cmap='viridis',
        norm=plt.cm.colors.LogNorm()
    )
    plt.colorbar(heatmap, label='Execution Time (ms)')
    plt.xlabel('Middle Dimension', fontsize=12)
    plt.ylabel('Sequence Length', fontsize=12)
    plt.title('Experiment 2: Execution Time for Various Sequence Lengths and Middle Dimensions', fontsize=14)
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xticks(middle_dims, [str(x) for x in middle_dims], rotation=45)
    plt.yticks(seq_lengths, [str(x) for x in seq_lengths])
    plt.tight_layout()
    plt.savefig('results/experiment2_heatmap.png', dpi=300)
    
    # Plot Experiment 2 - Line plot for different middle dimensions
    plt.figure(figsize=(12, 8))
    for mid_dim in middle_dims:
        subset = exp2_df[exp2_df['middle_dim'] == mid_dim]
        plt.plot(subset['seq_len'], subset['time_ms'], 'o-', linewidth=2, label=f'middle_dim={mid_dim}')
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Experiment 2: Execution Time vs. Sequence Length for Different Middle Dimensions', fontsize=14)
    plt.legend(fontsize=10)
    plt.xticks(seq_lengths, [str(x) for x in seq_lengths], rotation=45)
    plt.tight_layout()
    plt.savefig('results/experiment2_lines.png', dpi=300)
    
    # Plot Experiment 2 - Line plot for different sequence lengths
    plt.figure(figsize=(12, 8))
    for s_len in seq_lengths:
        subset = exp2_df[exp2_df['seq_len'] == s_len]
        plt.plot(subset['middle_dim'], subset['time_ms'], 'o-', linewidth=2, label=f'seq_len={s_len}')
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Middle Dimension', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Experiment 2: Execution Time vs. Middle Dimension for Different Sequence Lengths', fontsize=14)
    plt.legend(fontsize=10)
    plt.xticks(middle_dims, [str(x) for x in middle_dims], rotation=45)
    plt.tight_layout()
    plt.savefig('results/experiment2_lines_by_seq.png', dpi=300)
    
    print("Plots saved to the 'results' directory.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. This script requires a GPU.")
        exit(1)
    
    # Print GPU info
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Run all experiments
    start_time = time.time()
    run_all_experiments()
    total_time = time.time() - start_time
    
    print(f"\nAll experiments completed in {total_time:.2f} seconds")
    
    # Plot the results
    plot_results()
    
    print("Done!")