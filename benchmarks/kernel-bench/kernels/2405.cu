#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void coalesced_warp_matmul_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            const int M, const int N, const int K) {
    // Each warp processes one row of output
    const unsigned int lane = threadIdx.x;
    const unsigned int warpId = threadIdx.y;
    const int row = blockIdx.y * blockDim.y + warpId;
    
    if (row >= M) return;
    
    // Each thread processes multiple consecutive elements
    const int elementsPerThread = (K + 31) / 32;
    float thread_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Base pointers for current row
    const float* row_A = A + row * K;
    
    // Process 4 columns of output per iteration
    for (int col = blockIdx.x * 4; col < N; col += gridDim.x * 4) {
        float col_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        // Process this row in chunks of 32 elements (1 element per thread)
        for (int k_base = 0; k_base < K; k_base += 32) {
            // Load 1 element per thread from A
            float a_val = 0.0f;
            int k_idx = k_base + lane;
            if (k_idx < K) {
                a_val = row_A[k_idx];
            }

            // Load corresponding elements from B for each output column
            for (int c = 0; c < 4 && col + c < N; c++) {
                float b_val = 0.0f;
                if (k_idx < K) {
                    b_val = B[(col + c) * K + k_idx];
                }
                
                // Compute partial dot products
                col_sums[c] += a_val * b_val;
            }
        }

        // Warp-level reduction for each column
        unsigned int mask = 0xffffffff;
        #pragma unroll
        for (int c = 0; c < 4 && col + c < N; c++) {
            float sum = col_sums[c];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(mask, sum, offset);
            }
            
            // Write result
            if (lane == 0) {
                C[row * N + col + c] = sum;
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Launch configuration
    const int warpsPerBlock = 8;
    const int threadsPerWarp = 32;
    dim3 block(threadsPerWarp, warpsPerBlock);
    
    // Each block processes warpsPerBlock rows and multiple columns
    // Adjust grid size based on matrix dimensions
    int grid_y = (M + warpsPerBlock - 1) / warpsPerBlock;
    int grid_x = (N + 127) / 128; // Process 128 columns per block (32 threads * 4 columns per thread)
    dim3 grid(grid_x, grid_y);

    coalesced_warp_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced warp-level matrix multiplication with transposed B (CUDA)");
}