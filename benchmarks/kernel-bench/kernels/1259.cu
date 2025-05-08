#include <torch/extension.h>
#include <cuda_runtime.h>

// Define tile sizes
#define TILE_K 32
#define TILE_L 16
#define BLOCK_SIZE 256

__global__ void optimized_einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Shared memory for tiling
    __shared__ float A_shared[TILE_L];
    __shared__ float B_shared[TILE_L][TILE_K];
    __shared__ float B_shared[TILE_L][TILE_K];
    
    // Calculate global position
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int elements_per_block = blockDim.x;
    int global_idx = bid * elements_per_block + tid;
    
    // Calculate indices
    int k = global_idx % K;
    int remainder = global_idx / K;
    int j = remainder % J;
    remainder /= J;
    int i = remainder % I;
    int b = remainder / I;
    
    // Early exit for invalid threads
    if (b >= BATCH || i >= I || j >= J || k >= K) return;
    
    float sum = 0.0f;
    
    // Process tiles along L dimension
    for (int l_tile = 0; l_tile < L; l_tile += TILE_L) {
        // Collaborative loading of A and B tiles into shared memory
        if (tid < TILE_L) {
            int l = l_tile + tid;
            if (l < L) {
                A_shared[tid] = A[b*I*J*L + i*J*L + j*L + l];
                // Each thread loads multiple elements of B for better memory coalescing
                #pragma unroll
                for (int k_local = 0; k_local < TILE_K; k_local += BLOCK_SIZE/TILE_L) {
                    int k_idx = k_local + (tid * BLOCK_SIZE/TILE_L);
                    if (k_idx < TILE_K && (l < L) && (k_idx + blockIdx.y * TILE_K < K)) {
                        B_shared[tid][k_idx] = B[l*K + k_idx + blockIdx.y * TILE_K];
                    }
                }
            }
        }
        __syncthreads();
        
        // Compute partial sums for this tile
        #pragma unroll
        for (int l_local = 0; l_local < TILE_L && (l_tile + l_local) < L; ++l_local) {
            sum += A_shared[l_local] * B_shared[l_local][k % TILE_K];
        }
        __syncthreads();
    }
    
    // Write result
    C[b*I*J*K + i*J*K + j*K + k] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in L");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(
        (BATCH * I * J * K + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (K + TILE_K - 1) / TILE_K
    );
    
    optimized_einsum_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 4D tensor-matrix multiplication (CUDA)");
}