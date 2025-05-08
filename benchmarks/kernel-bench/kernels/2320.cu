#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__global__ void matmul_coalesced_warp_kernel(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            float* __restrict__ C, 
                                            const int M, const int N, const int K) {
    // Each thread processes multiple elements for better memory coalescing
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Calculate starting row for this thread block
    const int block_row = blockIdx.y * (BLOCK_SIZE / WARP_SIZE) * ELEMENTS_PER_THREAD;
    
    // Each warp handles multiple rows
    const int row = block_row + wid * ELEMENTS_PER_THREAD;
    const int col = blockIdx.x * WARP_SIZE + lane;

    // Registers to accumulate results
    float sums[ELEMENTS_PER_THREAD] = {0.0f};

    // Process K dimension in chunks
    if (col < N) {
        for (int k = 0; k < K; k++) {
            // Load B value once and reuse for multiple rows
            const float b_val = __ldg(&B[col * K + k]);
            
            // Process multiple rows per thread
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
                if (row + i < M) {
                    const float a_val = __ldg(&A[(row + i) * K + k]);
                    sums[i] += a_val * b_val;
                }
            }
        }

        // Store results - coalesced write
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            if (row + i < M) {
                C[(row + i) * N + col] = sums[i];
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Calculate grid dimensions
    const int rows_per_block = (BLOCK_SIZE / WARP_SIZE) * ELEMENTS_PER_THREAD;
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + WARP_SIZE - 1) / WARP_SIZE,
              (M + rows_per_block - 1) / rows_per_block);

    matmul_coalesced_warp_kernel<<<grid, block>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using coalesced memory access (CUDA)");
}