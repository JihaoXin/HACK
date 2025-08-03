# Matrix Multiplication Performance Analysis

This report analyzes the performance of two matrix multiplication experiments using PyTorch with bfloat16 precision on an NVIDIA A100-SXM4-40GB GPU.

## Experiment Setup

- **Hardware**: NVIDIA A100-SXM4-40GB
- **Precision**: bfloat16
- **Model Dimension**: 4096 (fixed)
- **Sequence Lengths**: [64, 128, 256, 512, 1024, 2048, 4096]
- **Middle Dimensions**: [64, 128, 256, 512, 1024, 2048, 4096]

## Experiment 1: Two Consecutive Matrix Multiplications

Experiment 1 measures the performance of two consecutive matrix multiplications:
- First multiplication: [seq_len, model_dim] @ [model_dim, model_dim]
- Second multiplication: [seq_len, model_dim] @ [model_dim, seq_len]

### Key Findings

1. **Scaling Behavior**: The execution time scales roughly linearly with sequence length when plotted on a log-log scale.
2. **Performance Range**: Execution times ranged from ~0.06ms for seq_len=64 to ~1.33ms for seq_len=4096.
3. **Doubling Relationship**: When the sequence length doubles, execution time increases by approximately 60-100%.

### Sequence Length Impact

| Sequence Length | Time (ms) | Scaling Factor |
|-----------------|-----------|----------------|
| 64              | 0.062     | -              |
| 128             | 0.076     | 1.22x          |
| 256             | 0.108     | 1.42x          |
| 512             | 0.161     | 1.49x          |
| 1024            | 0.350     | 2.17x          |
| 2048            | 0.672     | 1.92x          |
| 4096            | 1.334     | 1.99x          |

## Experiment 2: Three Matrix Multiplications with Middle Dimension

Experiment 2 measures the performance of three matrix multiplications with a variable middle dimension:
- Left branch: [seq_len, model_dim] @ [model_dim, middle_dim]
- Right branch: [middle_dim, model_dim] @ [model_dim, seq_len]
- Final multiplication: [seq_len, middle_dim] @ [middle_dim, seq_len]

### Key Findings

1. **Middle Dimension Impact**: Increasing the middle dimension significantly impacts performance, especially at larger sequence lengths.
2. **Optimal Middle Dimension**: Smaller middle dimensions (64-256) generally provide the best performance for all sequence lengths.
3. **Performance Range**: Execution times ranged from ~0.054ms (seq_len=64, middle_dim=64) to ~1.992ms (seq_len=4096, middle_dim=4096).
4. **Scaling Behavior**: For smaller sequence lengths (64-256), the middle dimension has less impact. As sequence length increases, the impact of middle dimension grows significantly.

### Middle Dimension Impact

For sequence length = 4096:
| Middle Dimension | Time (ms) | Scaling Factor |
|------------------|-----------|----------------|
| 64               | 0.141     | -              |
| 128              | 0.176     | 1.25x          |
| 256              | 0.228     | 1.30x          |
| 512              | 0.351     | 1.54x          |
| 1024             | 0.725     | 2.07x          |
| 2048             | 1.149     | 1.59x          |
| 4096             | 1.992     | 1.73x          |

## Comparison Between Experiments

1. **Efficiency**: For the same sequence length, Experiment 2 with a small middle dimension (64-256) is faster than Experiment 1, despite performing three matrix multiplications instead of two.
2. **Bottleneck Analysis**: The major performance bottleneck in Experiment 1 appears to be the first multiplication with the full model_dim.
3. **Optimization Strategy**: Using a smaller middle dimension effectively reduces the computational complexity by decreasing the size of intermediate matrices.

## Recommendations

1. **Use Small Middle Dimensions**: When implementing architectures with factorized matrices, use smaller middle dimensions (around 64-256) for optimal performance.
2. **Sequence Length Selection**: For interactive applications requiring low latency, keep sequence lengths below 1024 if possible.
3. **Scaling Strategy**: As sequence length increases, the benefit of using smaller middle dimensions becomes more pronounced. Adapt the middle dimension based on sequence length for optimal performance.

## Conclusion

The experiments demonstrate that factorizing large matrix multiplications using smaller middle dimensions can significantly improve performance, especially for longer sequence lengths. This approach is particularly relevant for transformer models where attention mechanisms involve large matrix multiplications that can be factorized for better efficiency. 