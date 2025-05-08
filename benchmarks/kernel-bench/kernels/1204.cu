#include <torch/extension.h>
#include <cuda_runtime.h>

// Shared memory tile sizes
#define TILE_SIZE_L 32
#define TILE_SIZE_K 32

__global__ void einsum_kernel_optimized_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    __shared__ float A_tile[TILE_SIZE_L];
    __shared__ float B_tile[TILE_SIZE_L][TILE_SIZE_K];
    
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = BATCH * I * J * K;
    if (global_idx >= total_elements) return;

    int k = global_idx % K;
    int remainder = global_idx / K;
    int j = remainder % J;
    remainder /= J;
    int i = remainder % I;
    int b = remainder / I;

    float sum = 0.0f;
    
    // Process tiles
    for (int tile = 0; tile < L; tile += TILE_SIZE_L) {
        // Load A tile cooperatively
        if (threadIdx.x < TILE_SIZE_L && (tile + threadIdx.x) < L) {
            int a_offset = b * I*J*L + i*J*L + j*L + tile + threadIdx.x;
            A_tile[threadIdx.x] = __ldg(&A[a_offset]);
        }
        
        // Load B tile cooperatively
        if (threadIdx.x < TILE_SIZE_L && (tile + threadIdx.x) < L) {
            int b_offset = (tile + threadIdx.x)*K + k;
            B_tile[threadIdx.x][k % TILE_SIZE_K] = __ldg(&B[b_offset]);
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int l = 0; l < TILE_SIZE_L && (tile + l) < L; ++l) {
            sum += A_tile[l] * B_tile[l][k % TILE_SIZE_K];
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
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    einsum_kernel_optimized_tiled<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled 4D tensor-matrix multiplication with shared memory (CUDA)");
}