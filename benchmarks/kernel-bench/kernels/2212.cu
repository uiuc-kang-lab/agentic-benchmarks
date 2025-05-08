#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_DIM 16
#define BLOCK_ROWS 16
#define ALIGN_SIZE 32  // For memory alignment

__global__ void coalescedMatMulKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     const int K, const int M, const int N) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[TILE_DIM][TILE_DIM + 1];  // +1 padding to avoid bank conflicts
    __shared__ float Bs[TILE_DIM][TILE_DIM + 1];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Index for writing output in a coalesced manner
    int row = bx * TILE_DIM + tx;
    int col = by * TILE_DIM + ty;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_DIM - 1) / TILE_DIM; ++tile) {
        // Collaborative loading of A and B tiles into shared memory
        // Ensuring coalesced global memory access pattern
        if (row < M && (tile * TILE_DIM + ty) < K) {
            // Load A in a coalesced manner - consecutive threads read consecutive elements
            As[tx][ty] = __ldg(&A[(tile * TILE_DIM + ty) * M + row]);
        } else {
            As[tx][ty] = 0.0f;
        }
        
        if (col < N && (tile * TILE_DIM + tx) < K) {
            // Load B in a coalesced manner
            Bs[tx][ty] = __ldg(&B[(tile * TILE_DIM + tx) * N + col]);
        } else {
            Bs[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot products
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += As[tx][k] * Bs[k][ty];
        }
        
        __syncthreads();
    }
    
    // Write result in a coalesced manner
    if (row < M && col < N) {
        // Ensure coalesced writes by having consecutive threads write to consecutive memory locations
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");
    
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");
    int N = B.size(1);
    
    // Pad dimensions to align memory accesses
    int M_aligned = ((M + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    int N_aligned = ((N + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
    
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));
    
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();
    
    // Configure kernel launch parameters
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((M + TILE_DIM - 1) / TILE_DIM,
                   (N + TILE_DIM - 1) / TILE_DIM);
    
    // Launch kernel with aligned dimensions
    coalescedMatMulKernel<<<numBlocks, threadsPerBlock>>>(
        A_ptr, B_ptr, C_ptr, K, M, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced matrix multiplication (CUDA)");
}