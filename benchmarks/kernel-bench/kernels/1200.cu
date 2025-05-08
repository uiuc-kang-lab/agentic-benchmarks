#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void einsum_kernel_shared(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];
    
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= BATCH * I * J * K) return;

    int k = global_idx % K;
    int remainder = global_idx / K;
    int j = remainder % J;
    remainder /= J;
    int i = remainder % I;
    int b = remainder / I;

    float sum = 0.0f;
    
    // Process B matrix in tiles
    for (int tile = 0; tile < (L + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Collaborative loading of B tile into shared memory
        if (threadIdx.x < TILE_SIZE) {
            for (int ty = threadIdx.x; ty < TILE_SIZE; ty += blockDim.x) {
                int l = tile * TILE_SIZE + ty;
                if (l < L && k < K) {
                    B_shared[ty][threadIdx.x] = B[l * K + k];
                } else {
                    B_shared[ty][threadIdx.x] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Compute partial sum using the tile
        #pragma unroll 8
        for (int l_offset = 0; l_offset < TILE_SIZE; ++l_offset) {
            int l = tile * TILE_SIZE + l_offset;
            if (l < L) {
                int a_offset = b * I*J*L + i*J*L + j*L + l;
                sum += A[a_offset] * B_shared[l_offset][k % TILE_SIZE];
            }
        }
        __syncthreads();
    }
    
    C[global_idx] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    int total_elements = BATCH * I * J * K;
    
    // Ensure thread block size is multiple of warp size (32) for optimal performance
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    einsum_kernel_shared<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with shared memory tiling (CUDA)");
}