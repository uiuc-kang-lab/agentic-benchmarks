#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

__global__ void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int batch_idx = by / (I * J);
    int ij_idx = by % (I * J);
    int i = ij_idx / J;
    int j = ij_idx % J;
    int k = bx * BLOCK_SIZE + tx;
    
    if (batch_idx >= BATCH || i >= I || j >= J || k >= K) return;
    
    float sum = 0.0f;
    
    for (int l_tile = 0; l_tile < L; l_tile += TILE_SIZE) {
        if ((ty + l_tile) < L && k < K) {
            B_shared[ty][tx] = (ty + l_tile) < L ? B[(ty + l_tile) * K + k] : 0.0f;
        } else {
            B_shared[ty][tx] = 0.0f;
        }
        __syncthreads();
        
        for (int l_local = 0; l_local < TILE_SIZE && (l_local + l_tile) < L; ++l_local) {
            sum += A[batch_idx * I * J * L + i * J * L + j * L + (l_local + l_tile)] * 
                   B_shared[l_local][tx];
        }
        __syncthreads();
    }
    
    if (batch_idx < BATCH && i < I && j < J && k < K) {
        C[batch_idx * I * J * K + i * J * K + j * K + k] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    
    dim3 threads(BLOCK_SIZE, TILE_SIZE);
    dim3 blocks((K + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                BATCH * I * J);
    
    einsum_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication (CUDA)");
}