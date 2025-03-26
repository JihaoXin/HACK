import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch

def load_singular_values(file_path):
    svals = torch.load(file_path).cpu().numpy()
    # Ensure singular values are sorted in descending order
    return np.sort(svals)[::-1]

def plot_singular_values(svals, title, save_path):
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot singular values
    plt.semilogy(range(1, len(svals) + 1), svals, 'b-', label='Singular Values')
    
    # Calculate and display some statistics
    total_energy = np.sum(svals)
    cumsum = np.cumsum(svals)
    energy_ratio = cumsum / total_energy
    
    # Find indices where energy ratio reaches certain thresholds
    idx_50 = np.where(energy_ratio >= 0.5)[0][0] + 1
    idx_90 = np.where(energy_ratio >= 0.9)[0][0] + 1
    idx_95 = np.where(energy_ratio >= 0.95)[0][0] + 1
    
    # Add vertical lines for these thresholds
    plt.axvline(x=idx_50, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=idx_90, color='g', linestyle='--', alpha=0.5)
    plt.axvline(x=idx_95, color='y', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title(f'Singular Values Distribution - {title}\n'
              f'50% energy at {idx_50}, 90% at {idx_90}, 95% at {idx_95}')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def plot_average_singular_values(all_svals_dict, save_path):
    plt.figure(figsize=(12, 8))
    
    colors = {'down_proj_weight': 'b', 'gate_proj_weight': 'r', 'up_proj_weight': 'g'}
    
    for dir_name, svals_list in all_svals_dict.items():
        if not svals_list:
            continue
            
        # Convert list of arrays to 2D array
        svals_array = np.stack(svals_list)
        
        # Calculate mean and std
        mean_svals = np.mean(svals_array, axis=0)
        std_svals = np.std(svals_array, axis=0)
        
        # Calculate energy ratios for mean values
        total_energy = np.sum(mean_svals)
        cumsum = np.cumsum(mean_svals)
        energy_ratio = cumsum / total_energy
        
        # Find indices where energy ratio reaches certain thresholds
        idx_50 = np.where(energy_ratio >= 0.5)[0][0] + 1
        idx_90 = np.where(energy_ratio >= 0.9)[0][0] + 1
        idx_95 = np.where(energy_ratio >= 0.95)[0][0] + 1
        
        # Plot mean with error bands
        x = range(1, len(mean_svals) + 1)
        plt.semilogy(x, mean_svals, colors[dir_name], label=f'{dir_name} (50% at {idx_50})', alpha=0.8)
        plt.fill_between(x, mean_svals - std_svals, mean_svals + std_svals, 
                        color=colors[dir_name], alpha=0.2)
        
        # Add vertical lines for these thresholds
        plt.axvline(x=idx_50, color=colors[dir_name], linestyle='--', alpha=0.3)
        plt.axvline(x=idx_90, color=colors[dir_name], linestyle=':', alpha=0.3)
        plt.axvline(x=idx_95, color=colors[dir_name], linestyle='-.', alpha=0.3)
    
    plt.xlabel('Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title('Average Singular Values Distribution by Layer Type')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    # Create output directory if it doesn't exist
    output_dir = Path('singular_value_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Dictionary to store all singular values by directory
    all_svals = {
        'down_proj_weight': [],
        'gate_proj_weight': [],
        'up_proj_weight': []
    }
    
    # Process each weight matrix
    for dir_name in all_svals.keys():
        dir_path = Path('data') / dir_name
        if not dir_path.exists():
            print(f"Directory {dir_path} does not exist")
            continue
            
        # Get all .pt files in the directory
        pt_files = list(dir_path.glob('*.pt'))
        
        for file_path in pt_files:
            print(f"Processing {file_path}")
            try:
                svals = load_singular_values(file_path)
                all_svals[dir_name].append(svals)
                
                # Create individual plot
                title = f"{dir_name} - {file_path.stem}"
                save_path = output_dir / f"{dir_name}_{file_path.stem}_svals.png"
                plot_singular_values(svals, title, save_path)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Create summary plot
    plot_average_singular_values(all_svals, output_dir / 'summary_svals.png')

if __name__ == "__main__":
    main() 