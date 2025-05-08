#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    __shared__ float A_tile[TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_idx >= BATCH * I * J * K) return;
    
    int k = global_idx % K;
    int remainder = global_idx / K;
    int j = remainder % J;
    remainder /= J;
    int i = remainder % I;
    int b = remainder / I;
    
    float sum = 0.0f;
    
    // Initialize sum to 0 to avoid undefined behavior
    C[global_idx] = 0.0f;
    
    for(int l_base = 0; l_base < L; l_base += TILE_SIZE) {
        // Collaborative loading of tiles
        if (tid + l_base < L) {
            // Fix indexing for A and B matrices
            A_tile[tid] = A[((b * I + i) * J + j) * L + (l_base + tid)];
            B_tile[tid] = B[(l_base + tid) * K + k];
        }
        
        __syncthreads();
        
        // Only process elements within valid range
        int tile_size = min(TILE_SIZE, L - l_base);
        #pragma unroll
        for(int l = 0; l < tile_size; ++l) {
            sum += A_tile[l] * B_tile[l];
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
    
    int threads = TILE_SIZE;
    int blocks = (total_elements + threads - 1) / threads;
    
    einsum_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        BATCH, I, J, L, K);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication (CUDA)");
}