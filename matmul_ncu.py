import torch
WARMUP_ITERATIONS = 0
def experiment1():
    """
    Two consecutive matmuls, size [100,4k]@[4k,4k]@[4k,100]
    """
    # Initialize matrices on GPU
    A = torch.rand(100, 4000, dtype=torch.bfloat16, device='cuda')
    B = torch.rand(4000, 4000, dtype=torch.bfloat16, device='cuda')
    C = torch.rand(4000, 100, dtype=torch.bfloat16, device='cuda')
    
    # The actual computation to be profiled
    # torch.cuda.synchronize()
    temp = A @ B
    result = temp @ C
    # torch.cuda.synchronize()
    return result

def experiment2(middle_dim=300):
    """
    Three matmuls: ([100,4k]@[4k,300])@([300,4k]@[4k,100])
    """
    # Initialize matrices on GPU
    A = torch.rand(100, 4000, dtype=torch.bfloat16, device='cuda')
    B = torch.rand(4000, middle_dim, dtype=torch.bfloat16, device='cuda')
    C = torch.rand(middle_dim, 4000, dtype=torch.bfloat16, device='cuda')
    D = torch.rand(4000, 100, dtype=torch.bfloat16, device='cuda')

    # The actual computation to be profiled
    # torch.cuda.synchronize()
    left = A @ B
    right = C @ D
    result = left @ right
    # torch.cuda.synchronize()
    return result

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. This script requires a GPU.")
        exit(1)
    
    print("Running experiment 1...")
    result1 = experiment1()
    
    print("Running experiment 2...")
    result2 = experiment2(middle_dim=512)
    
    print("Done. Use the following commands to profile with NCU:")
    print("\nFor experiment 1:")
    print("ncu --metrics sm__cycles_active.avg.pct_of_peak_sustained_elapsed,sm__cycles_elapsed.avg.pct_of_peak_sustained_elapsed python -c \"import matmul_ncu; matmul_ncu.experiment1()\"")
    print("\nFor experiment 2:")
    print("ncu --metrics sm__cycles_active.avg.pct_of_peak_sustained_elapsed,sm__cycles_elapsed.avg.pct_of_peak_sustained_elapsed python -c \"import matmul_ncu; matmul_ncu.experiment2()\"") 