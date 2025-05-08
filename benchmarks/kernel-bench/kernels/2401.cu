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
    for (int col = blockIdx.x * blockDim.x * 4 + lane; col < N; col += blockDim.x * gridDim.x * 4) {
        float col_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        // Process this row in chunks of 128 elements (4 elements per thread * 32 threads)
        for (int k_base = 0; k_base < K; k_base += 128) {
            // Load 4 consecutive elements per thread from A
            float4 a_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            int k_idx = k_base + lane * 4;
            if (k_idx < K) {
                if (k_idx + 3 < K) {
                    a_val = *reinterpret_cast<const float4*>(row_A + k_idx);
                } else {
                    a_val.x = row_A[k_idx];
                    a_val.y = (k_idx + 1 < K) ? row_A[k_idx + 1] : 0.0f;
                    a_val.z = (k_idx + 2 < K) ? row_A[k_idx + 2] : 0.0f;
                    a_val.w = 0.0f;
                }
            }

            // Load corresponding elements from B for each output column
            for (int c = 0; c < 4 && col + c < N; c++) {
                float4 b_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (k_idx < K) {
                    const float* col_B = B + (col + c) * K + k_idx;
                    if (k_idx + 3 < K) {
                        b_val.x = col_B[0];
                        b_val.y = col_B[1];
                        b_val.z = col_B[2];
                        b_val.w = col_B[3];
                    } else {
                        b_val.x = col_B[0];
                        b_val.y = (k_idx + 1 < K) ? col_B[1] : 0.0f;
                        b_val.z = (k_idx + 2 < K) ? col_B[2] : 0.0f;
                    }
                }
                
                // Compute partial dot products
                col_sums[c] += a_val.x * b_val.x + a_val.y * b_val.y +
                              a_val.z * b_val.z + a_val.w * b_val.w;
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