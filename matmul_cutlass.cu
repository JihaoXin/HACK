#include <iostream>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/command_line.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <chrono>  // For timing

// Define the precision to use
using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAccumulator = float;

// For the original experiments, we can also define the bf16 types
using ElementBF16 = cutlass::bfloat16_t;
using ElementBF16Acc = float;

void checkCUDA(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at " << file << ":" << line 
                  << " " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(A) checkCUDA(A, __FILE__, __LINE__);

// Define the CUTLASS GEMM type for our computation (simpler version with float)
using CutlassGemmF32 = cutlass::gemm::device::Gemm<
    ElementA,                                // ElementA
    cutlass::layout::RowMajor,               // LayoutA
    ElementB,                                // ElementB
    cutlass::layout::RowMajor,               // LayoutB
    ElementC,                                // ElementC
    cutlass::layout::RowMajor,               // LayoutC
    ElementAccumulator,                      // ElementAccumulator
    cutlass::arch::OpClassSimt,              // tag indicating Tensor Cores
    cutlass::arch::Sm80                      // tag indicating target architecture
>;

// Define the CUTLASS GEMM type for BF16 computation
using CutlassGemmBF16 = cutlass::gemm::device::Gemm<
    ElementBF16,                             // ElementA
    cutlass::layout::RowMajor,               // LayoutA
    ElementBF16,                             // ElementB
    cutlass::layout::RowMajor,               // LayoutB
    ElementBF16,                             // ElementC
    cutlass::layout::RowMajor,               // LayoutC
    ElementBF16Acc,                          // ElementAccumulator
    cutlass::arch::OpClassSimt,              // Use SIMT operations instead of Tensor Cores for debugging
    cutlass::arch::Sm80                      // tag indicating target architecture
>;

// Utility function to measure execution time
double measure_performance(std::function<void()> func, int num_iterations = 10) {
    // Warmup run
    func();
    
    // Synchronize before timing
    cudaDeviceSynchronize();
    
    // Measure time
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        func();
    }
    
    // Synchronize after timing
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    // Return average time per iteration in milliseconds
    return elapsed.count() / num_iterations;
}

// Simple matrix multiplication test
void simple_gemm() {
    std::cout << "Running Simple GEMM with CUTLASS..." << std::endl;
    
    // Define the matrix problem size
    const int M = 128;
    const int N = 128;
    const int K = 128;
    
    // Allocate tensors
    cutlass::HostTensor<ElementA, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<ElementB, cutlass::layout::RowMajor> B({K, N});
    cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> C({M, N});
    cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> D({M, N});
    
    // Fill tensors with data
    cutlass::reference::host::TensorFillRandomUniform(
        A.host_view(), 
        0,      // seed
        1.0f,   // max
        0.0f    // min
    );
    
    cutlass::reference::host::TensorFillRandomUniform(
        B.host_view(), 
        1,      // seed
        1.0f,   // max
        0.0f    // min
    );
    
    cutlass::reference::host::TensorFill(C.host_view(), 0.0f);
    cutlass::reference::host::TensorFill(D.host_view(), 0.0f);
    
    // Copy data to device
    A.sync_device();
    B.sync_device();
    C.sync_device();
    D.sync_device();
    
    // Initialize GEMM operator
    CutlassGemmF32 gemm_op;
    
    // Create arguments for GEMM operation
    typename CutlassGemmF32::Arguments args{
        {M, N, K},                  // problem size (m, n, k)
        A.device_ref(),             // pointer to A on device
        B.device_ref(),             // pointer to B on device
        C.device_ref(),             // pointer to C on device
        D.device_ref(),             // pointer to D on device
        {1.0f, 0.0f}                // alpha, beta
    };
    
    // Launch GEMM
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize before GEMM failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    cutlass::Status status = gemm_op(args);
    
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize after GEMM failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return;
    }
    
    // Copy results back to host for verification (optional)
    D.sync_host();
    
    // Verify the first few elements (optional)
    std::cout << "First few elements of the result matrix:" << std::endl;
    for (int i = 0; i < std::min(5, M); ++i) {
        for (int j = 0; j < std::min(5, N); ++j) {
            std::cout << D.host_ref().at({i, j}) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Simple GEMM completed successfully." << std::endl;
}

// Experiment 1 with BF16: Two consecutive matmuls [100,4k]@[4k,4k]@[4k,100]
void experiment1(bool measure_perf = false) {
    std::cout << "Running Experiment 1 with CUTLASS (BF16)..." << std::endl;
    
    const int M = 100;
    const int K1 = 4000;
    const int N1 = 4000;
    const int K2 = 4000;
    const int N2 = 100;
    
    // Allocate tensors
    cutlass::HostTensor<ElementBF16, cutlass::layout::RowMajor> A({M, K1});
    cutlass::HostTensor<ElementBF16, cutlass::layout::RowMajor> B({K1, N1});
    cutlass::HostTensor<ElementBF16, cutlass::layout::RowMajor> C({K2, N2});
    cutlass::HostTensor<ElementBF16, cutlass::layout::RowMajor> Temp({M, N1});
    cutlass::HostTensor<ElementBF16, cutlass::layout::RowMajor> Result({M, N2});
    
    // Fill tensors with data - using proper tensor initialization
    cutlass::reference::host::TensorFill(A.host_view(), ElementBF16(0.0f));  // Initialize to zeros first
    cutlass::reference::host::TensorFill(B.host_view(), ElementBF16(0.0f));
    cutlass::reference::host::TensorFill(C.host_view(), ElementBF16(0.0f));
    cutlass::reference::host::TensorFill(Temp.host_view(), ElementBF16(0.0f));
    cutlass::reference::host::TensorFill(Result.host_view(), ElementBF16(0.0f));
    
    // Manually fill with random values
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K1; ++j) {
            A.host_ref().at({i, j}) = ElementBF16(0.1f * (float(rand()) / RAND_MAX));
        }
    }
    
    for (int i = 0; i < K1; ++i) {
        for (int j = 0; j < N1; ++j) {
            B.host_ref().at({i, j}) = ElementBF16(0.1f * (float(rand()) / RAND_MAX));
        }
    }
    
    for (int i = 0; i < K2; ++i) {
        for (int j = 0; j < N2; ++j) {
            C.host_ref().at({i, j}) = ElementBF16(0.1f * (float(rand()) / RAND_MAX));
        }
    }
    
    // Copy data to device
    A.sync_device();
    B.sync_device();
    C.sync_device();
    Temp.sync_device();
    Result.sync_device();
    
    // Initialize GEMM operators
    CutlassGemmBF16 gemm_op1;
    CutlassGemmBF16 gemm_op2;
    
    // Create arguments for the first GEMM (A @ B)
    typename CutlassGemmBF16::Arguments args_1{
        {M, N1, K1},                // problem size (m, n, k)
        A.device_ref(),             // pointer to A on device
        B.device_ref(),             // pointer to B on device
        Temp.device_ref(),          // pointer to C on device
        Temp.device_ref(),          // pointer to D on device
        {ElementBF16Acc(1.0f), ElementBF16Acc(0.0f)}  // alpha, beta
    };
    
    // Create arguments for the second GEMM (Temp @ C)
    typename CutlassGemmBF16::Arguments args_2{
        {M, N2, K2},                // problem size (m, n, k)
        Temp.device_ref(),          // pointer to A on device
        C.device_ref(),             // pointer to B on device
        Result.device_ref(),        // pointer to C on device
        Result.device_ref(),        // pointer to D on device
        {ElementBF16Acc(1.0f), ElementBF16Acc(0.0f)}  // alpha, beta
    };
    
    if (measure_perf) {
        // Define the GEMM execution functions for timing
        auto gemm1_func = [&]() {
            gemm_op1(args_1);
        };
        
        auto gemm2_func = [&]() {
            gemm_op2(args_2);
        };
        
        // Run both GEMMs once to verify they work
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        cutlass::Status status = gemm_op1(args_1);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM 1 operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return;
        }
        
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        status = gemm_op2(args_2);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM 2 operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return;
        }
        
        // Measure performance
        double gemm1_time = measure_performance(gemm1_func);
        double gemm2_time = measure_performance(gemm2_func);
        double total_time = gemm1_time + gemm2_time;
        
        // Calculate FLOPS
        // Each GEMM operation: 2 * M * N * K
        double flops_gemm1 = 2.0 * M * N1 * K1;  // Multiply-add counts as 2 operations
        double flops_gemm2 = 2.0 * M * N2 * K2;
        
        double total_flops = flops_gemm1 + flops_gemm2;
        
        // Calculate GFLOPS (billions of floating-point operations per second)
        double gflops_gemm1 = (flops_gemm1 / gemm1_time) / 1e6;  // ms to seconds conversion
        double gflops_gemm2 = (flops_gemm2 / gemm2_time) / 1e6;
        double gflops_total = (total_flops / total_time) / 1e6;
        
        // Report performance metrics
        std::cout << "Performance metrics (BF16 precision):" << std::endl;
        std::cout << "  GEMM 1 [" << M << "x" << K1 << "] @ [" << K1 << "x" << N1 << "]:" << std::endl;
        std::cout << "    Time: " << gemm1_time << " ms" << std::endl;
        std::cout << "    GFLOPS: " << gflops_gemm1 << std::endl;
        std::cout << "  GEMM 2 [" << M << "x" << N1 << "] @ [" << K2 << "x" << N2 << "]:" << std::endl;
        std::cout << "    Time: " << gemm2_time << " ms" << std::endl;
        std::cout << "    GFLOPS: " << gflops_gemm2 << std::endl;
        std::cout << "  Total:" << std::endl;
        std::cout << "    Time: " << total_time << " ms" << std::endl;
        std::cout << "    GFLOPS: " << gflops_total << std::endl;
    } else {
        // Run without performance measurement
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize before GEMM 1 failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        cutlass::Status status = gemm_op1(args_1);
        
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize after GEMM 1 failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM 1 operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return;
        }
        
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize before GEMM 2 failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        status = gemm_op2(args_2);
        
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize after GEMM 2 failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM 2 operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return;
        }
    }
    
    std::cout << "Experiment 1 (BF16) completed successfully." << std::endl;
}


// Experiment 2 with BF16: Three matmuls ([100,4k]@[4k,300])@([300,4k]@[4k,100])
void experiment2(int middle_dim = 512, bool measure_perf = false) {
    std::cout << "Running Experiment 2 with CUTLASS (BF16, middle_dim = " << middle_dim << ")..." << std::endl;
    
    const int M = 100;
    const int K1 = 4000;
    const int N1 = middle_dim;
    const int M2 = middle_dim;
    const int K2 = 4000;
    const int N2 = 100;
    
    // Print the dimensions for debugging
    std::cout << "Matrix dimensions:" << std::endl;
    std::cout << "  A: " << M << "x" << K1 << std::endl;
    std::cout << "  B: " << K1 << "x" << N1 << std::endl;
    std::cout << "  C: " << M2 << "x" << K2 << std::endl; 
    std::cout << "  D: " << K2 << "x" << N2 << std::endl;
    std::cout << "  Left: " << M << "x" << N1 << std::endl;
    std::cout << "  Right: " << M2 << "x" << N2 << std::endl;
    std::cout << "  Final GEMM: [" << M << "x" << N1 << "] @ [" << M2 << "x" << N2 << "] = [" << M << "x" << N2 << "]" << std::endl;
    
    // Ensure that dimensions match for the final GEMM
    if (N1 != M2) {
        std::cerr << "Error: Matrix dimensions mismatch for final GEMM." << std::endl;
        std::cerr << "Inner dimensions must match: N1 (" << N1 << ") != M2 (" << M2 << ")" << std::endl;
        return;
    }
    
    // Allocate tensors
    cutlass::HostTensor<ElementBF16, cutlass::layout::RowMajor> A({M, K1});
    cutlass::HostTensor<ElementBF16, cutlass::layout::RowMajor> B({K1, N1});
    cutlass::HostTensor<ElementBF16, cutlass::layout::RowMajor> C({M2, K2});
    cutlass::HostTensor<ElementBF16, cutlass::layout::RowMajor> D({K2, N2});
    cutlass::HostTensor<ElementBF16, cutlass::layout::RowMajor> Left({M, N1});
    cutlass::HostTensor<ElementBF16, cutlass::layout::RowMajor> Right({M2, N2});
    cutlass::HostTensor<ElementBF16, cutlass::layout::RowMajor> Result({M, N2});
    
    // Fill tensors with data - using proper tensor initialization
    cutlass::reference::host::TensorFill(A.host_view(), ElementBF16(0.0f));  // Initialize to zeros first
    cutlass::reference::host::TensorFill(B.host_view(), ElementBF16(0.0f));
    cutlass::reference::host::TensorFill(C.host_view(), ElementBF16(0.0f));
    cutlass::reference::host::TensorFill(D.host_view(), ElementBF16(0.0f));
    cutlass::reference::host::TensorFill(Left.host_view(), ElementBF16(0.0f));
    cutlass::reference::host::TensorFill(Right.host_view(), ElementBF16(0.0f));
    cutlass::reference::host::TensorFill(Result.host_view(), ElementBF16(0.0f));
    
    // Manually fill with random values
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K1; ++j) {
            A.host_ref().at({i, j}) = ElementBF16(0.1f * (float(rand()) / RAND_MAX));
        }
    }
    
    for (int i = 0; i < K1; ++i) {
        for (int j = 0; j < N1; ++j) {
            B.host_ref().at({i, j}) = ElementBF16(0.1f * (float(rand()) / RAND_MAX));
        }
    }
    
    for (int i = 0; i < M2; ++i) {
        for (int j = 0; j < K2; ++j) {
            C.host_ref().at({i, j}) = ElementBF16(0.1f * (float(rand()) / RAND_MAX));
        }
    }
    
    for (int i = 0; i < K2; ++i) {
        for (int j = 0; j < N2; ++j) {
            D.host_ref().at({i, j}) = ElementBF16(0.1f * (float(rand()) / RAND_MAX));
        }
    }
    
    // Copy data to device
    A.sync_device();
    B.sync_device();
    C.sync_device();
    D.sync_device();
    Left.sync_device();
    Right.sync_device();
    Result.sync_device();
    
    // Initialize GEMM operators
    CutlassGemmBF16 gemm_op_left;
    CutlassGemmBF16 gemm_op_right;
    CutlassGemmBF16 gemm_op_final;
    
    // Create arguments for the first GEMM (A @ B -> Left)
    typename CutlassGemmBF16::Arguments args_left{
        {M, N1, K1},               // problem size (m, n, k)
        A.device_ref(),            // pointer to A on device
        B.device_ref(),            // pointer to B on device
        Left.device_ref(),         // pointer to C on device
        Left.device_ref(),         // pointer to D on device
        {ElementBF16Acc(1.0f), ElementBF16Acc(0.0f)}  // alpha, beta
    };
    
    // Create arguments for the second GEMM (C @ D -> Right)
    typename CutlassGemmBF16::Arguments args_right{
        {M2, N2, K2},              // problem size (m, n, k)
        C.device_ref(),            // pointer to A on device
        D.device_ref(),            // pointer to B on device
        Right.device_ref(),        // pointer to C on device
        Right.device_ref(),        // pointer to D on device
        {ElementBF16Acc(1.0f), ElementBF16Acc(0.0f)}  // alpha, beta
    };
    
    // Create arguments for the final GEMM (Left @ Right -> Result)
    // Note: The K dimension in the final GEMM should be N1 since we're multiplying [M,N1] by [N1,N2]
    typename CutlassGemmBF16::Arguments args_final{
        {M, N2, N1},               // problem size (m, n, k)
        Left.device_ref(),         // pointer to A on device
        Right.device_ref(),        // pointer to B on device
        Result.device_ref(),       // pointer to C on device
        Result.device_ref(),       // pointer to D on device
        {ElementBF16Acc(1.0f), ElementBF16Acc(0.0f)}  // alpha, beta
    };
    
    if (measure_perf) {
        // Define the GEMM execution functions for timing
        auto gemm_left_func = [&]() {
            gemm_op_left(args_left);
        };
        
        auto gemm_right_func = [&]() {
            gemm_op_right(args_right);
        };
        
        auto gemm_final_func = [&]() {
            gemm_op_final(args_final);
        };
        
        // Run each GEMM with error checking
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        std::cout << "Running GEMM Left..." << std::endl;
        cutlass::Status status = gemm_op_left(args_left);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM Left operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return;
        }
        
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        std::cout << "Running GEMM Right..." << std::endl;
        status = gemm_op_right(args_right);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM Right operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return;
        }
        
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        // Debug - print some values from Left and Right tensors
        Left.sync_host();
        Right.sync_host();
        
        std::cout << "Left tensor (first few elements):" << std::endl;
        for (int i = 0; i < std::min(3, M); ++i) {
            for (int j = 0; j < std::min(3, N1); ++j) {
                std::cout << float(Left.host_ref().at({i, j})) << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "Right tensor (first few elements):" << std::endl;
        for (int i = 0; i < std::min(3, M2); ++i) {
            for (int j = 0; j < std::min(3, N2); ++j) {
                std::cout << float(Right.host_ref().at({i, j})) << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "Running GEMM Final..." << std::endl;
        status = gemm_op_final(args_final);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM Final operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return;
        }
        
        // Measure performance
        double gemm_left_time = measure_performance(gemm_left_func);
        double gemm_right_time = measure_performance(gemm_right_func);
        double gemm_final_time = measure_performance(gemm_final_func);
        double total_time = gemm_left_time + gemm_right_time + gemm_final_time;
        
        // Calculate FLOPS
        // Each GEMM operation: 2 * M * N * K
        double flops_left = 2.0 * M * N1 * K1;  // Multiply-add counts as 2 operations
        double flops_right = 2.0 * M2 * N2 * K2;
        double flops_final = 2.0 * M * N2 * N1;
        
        double total_flops = flops_left + flops_right + flops_final;
        
        // Calculate GFLOPS (billions of floating-point operations per second)
        double gflops_left = (flops_left / gemm_left_time) / 1e6;  // ms to seconds conversion
        double gflops_right = (flops_right / gemm_right_time) / 1e6;
        double gflops_final = (flops_final / gemm_final_time) / 1e6;
        double gflops_total = (total_flops / total_time) / 1e6;
        
        // Report performance metrics
        std::cout << "Performance metrics (BF16 precision, middle_dim=" << middle_dim << "):" << std::endl;
        std::cout << "  GEMM Left [" << M << "x" << K1 << "] @ [" << K1 << "x" << N1 << "]:" << std::endl;
        std::cout << "    Time: " << gemm_left_time << " ms" << std::endl;
        std::cout << "    GFLOPS: " << gflops_left << std::endl;
        std::cout << "  GEMM Right [" << M2 << "x" << K2 << "] @ [" << K2 << "x" << N2 << "]:" << std::endl;
        std::cout << "    Time: " << gemm_right_time << " ms" << std::endl;
        std::cout << "    GFLOPS: " << gflops_right << std::endl;
        std::cout << "  GEMM Final [" << M << "x" << N1 << "] @ [" << M2 << "x" << N2 << "]:" << std::endl;
        std::cout << "    Time: " << gemm_final_time << " ms" << std::endl;
        std::cout << "    GFLOPS: " << gflops_final << std::endl;
        std::cout << "  Total:" << std::endl;
        std::cout << "    Time: " << total_time << " ms" << std::endl;
        std::cout << "    GFLOPS: " << gflops_total << std::endl;
    } else {
        // Run without performance measurement but with error checking
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize before GEMM Left failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        cutlass::Status status = gemm_op_left(args_left);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM Left operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return;
        }
        
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize after GEMM Left failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        status = gemm_op_right(args_right);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM Right operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return;
        }
        
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize after GEMM Right failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        status = gemm_op_final(args_final);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM Final operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return;
        }
        
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize after GEMM Final failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }
    }
    
    std::cout << "Experiment 2 (BF16) completed successfully." << std::endl;
}

// Print usage information
void print_usage() {
    std::cout << "Usage: ./matmul_cutlass [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  simple           Run simple GEMM test" << std::endl;
    std::cout << "  exp1             Run experiment 1 with BF16 precision" << std::endl;
    std::cout << "  exp2 <middle_dim> Run experiment 2 with BF16 precision" << std::endl;
    std::cout << "  benchmark        Run BF16 experiments with performance measurements" << std::endl;
    std::cout << "  all              Run all tests with BF16 precision" << std::endl;
    std::cout << "  help             Display this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  ./matmul_cutlass exp2 300  # Run experiment 2 with middle_dim=300 (BF16)" << std::endl;
    std::cout << "  ./matmul_cutlass benchmark # Run BF16 experiments with performance measurements" << std::endl;
    std::cout << "  ./matmul_cutlass all       # Run all tests with BF16 precision" << std::endl;
}

int main(int argc, char* argv[]) {
    // Check that CUDA is available
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "CUDA not available. This program requires a GPU." << std::endl;
        return 1;
    }
    
    // Default middle dimension for experiment 2
    int middle_dim = 512;
    
    // Parse command line arguments
    if (argc > 1) {
        std::string arg = argv[1];
        
        // Check if middle_dim is provided for experiment 2
        if ((arg == "exp2_bf16") && argc > 2) {
            try {
                middle_dim = std::stoi(argv[2]);
                if (middle_dim <= 0) {
                    std::cerr << "Error: middle_dim must be a positive integer" << std::endl;
                    print_usage();
                    return 1;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Invalid middle dimension: " << argv[2] << std::endl;
                print_usage();
                return 1;
            }
        }
        
        if (arg == "simple") {
            simple_gemm();
        }
        else if (arg == "exp1") {
            experiment1();
        }
        else if (arg == "exp2") {
            experiment2(middle_dim);
        }
        else if (arg == "benchmark") {
            std::cout << "===== Running BF16 benchmarks =====" << std::endl;
            experiment1(true);  // true enables performance measurement
            
            std::cout << "\n----- Testing different middle dimensions for Experiment 2 -----" << std::endl;
            // Start with smaller middle dimensions that are more likely to work
            for (int dim : {32, 64, 128, 256}) {
                std::cout << "\nWith middle_dim = " << dim << ":" << std::endl;
                experiment2(dim, true);  // Enable performance measurement
            }
        }
        else if (arg == "all") {
            experiment1();
            experiment2(middle_dim);
        }
        else if (arg == "help") {
            print_usage();
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }
    else {
        // Default behavior: run simple test
        std::cout << "Running with default configuration" << std::endl;
        simple_gemm();
    }
    
    return 0;
} 