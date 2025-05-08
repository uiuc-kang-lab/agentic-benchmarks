#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int batch_i = blockIdx.z / I;
    int idx_i = blockIdx.z % I;
    int j_block = blockIdx.y * blockDim.y;
    int k_block = blockIdx.x * blockDim.x;
    
    int j = j_block + ty;
    int k = k_block + tx;
    
    float sum = 0.0f;
    
    for (int l_block = 0; l_block < L; l_block += TILE_SIZE) {
        int l = l_block + tx;
        
        // Load A tile
        if (j < J && l < L) {
            A_shared[ty][tx] = __ldg(&A[batch_i * I*J*L + idx_i * J*L + j * L + l]);
        } else {
            A_shared[ty][tx] = 0.0f;
        }
        
        // Load B tile using new x position
        l = l_block + ty;
        if (l < L && k < K) {
            B_shared[ty][tx] = __ldg(&B[l * K + k]);
        } else {
            B_shared[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int l_tile = 0; l_tile < TILE_SIZE; ++l_tile) {
            sum += A_shared[ty][l_tile] * B_shared[l_tile][tx];
        }
        
        __syncthreads();
    }
    
    if (j < J && k < K) {
        int c_idx = batch_i * I*J*K + idx_i * J*K + j * K + k;
        C[c_idx] = sum;
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
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (K + TILE_SIZE - 1) / TILE_SIZE,
        (J + TILE_SIZE - 1) / TILE_SIZE,
        BATCH * I
    );
    
    einsum_kernel<<<blocks, threads>>>( 
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with shared memory tiling and __ldg optimization (CUDA)");
}
