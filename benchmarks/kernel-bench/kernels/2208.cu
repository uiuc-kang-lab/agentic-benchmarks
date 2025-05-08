#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_DIM 16
#define BLOCK_ROWS 16
#define VEC_SIZE 4  // Using float4 for vectorized loads
#define PADDING 2   // Add padding to reduce bank conflicts

__global__ void vectorizedTiledMatMulKernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           const int K, const int M, const int N) {
    // Padded shared memory to reduce bank conflicts
    __shared__ float As[TILE_DIM][TILE_DIM + PADDING];
    __shared__ float Bs[TILE_DIM][TILE_DIM + PADDING];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int row = bx * TILE_DIM + tx;
    const int col = by * TILE_DIM + ty;
    
    float sum = 0.0f;
    
    // Use float4 for vectorized loading
    float4 a_vec, b_vec;
    
    // Main loop over tiles
    for (int tile = 0; tile < (K + TILE_DIM - 1) / TILE_DIM; ++tile) {
        // Collaborative loading of A and B tiles into shared memory using vectorized loads
        if ((tile * TILE_DIM + ty) < K && row < M) {
            #pragma unroll 4
            for (int i = 0; i < TILE_DIM; i += VEC_SIZE) {
                if (i + VEC_SIZE <= TILE_DIM) {
                    // Load 4 elements at once using float4
                    const int offset = (tile * TILE_DIM + ty) * M + row;
                    if (offset % VEC_SIZE == 0) {  // Ensure aligned access
                        a_vec = *reinterpret_cast<const float4*>(&A[offset + i]);
                        As[ty][i] = a_vec.x;
                        As[ty][i+1] = a_vec.y;
                        As[ty][i+2] = a_vec.z;
                        As[ty][i+3] = a_vec.w;
                    } else {
                        // Fallback for unaligned access
                        As[ty][i] = A[offset + i];
                        As[ty][i+1] = A[offset + i+1];
                        As[ty][i+2] = A[offset + i+2];
                        As[ty][i+3] = A[offset + i+3];
                    }
                }
            }
        }
        
        if ((tile * TILE_DIM + tx) < K && col < N) {
            #pragma unroll 4
            for (int i = 0; i < TILE_DIM; i += VEC_SIZE) {
                if (i + VEC_SIZE <= TILE_DIM) {
                    const int offset = (tile * TILE_DIM + tx) * N + col;
                    if (offset % VEC_SIZE == 0) {  // Ensure aligned access
                        b_vec = *reinterpret_cast<const float4*>(&B[offset + i]);
                        Bs[tx][i] = b_vec.x;
                        Bs[tx][i+1] = b_vec.y;
                        Bs[tx][i+2] = b_vec.z;
                        Bs[tx][i+3] = b_vec.w;
                    } else {
                        // Fallback for unaligned access
                        Bs[tx][i] = B[offset + i];
                        Bs[tx][i+1] = B[offset + i+1];
                        Bs[tx][i+2] = B[offset + i+2];
                        Bs[tx][i+3] = B[offset + i+3];
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial dot products
        #pragma unroll 8
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += As[k][tx] * Bs[k][ty];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");
    
    const int K = A.size(0);
    const int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");
    const int N = B.size(1);
    
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));
    
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    
    vectorizedTiledMatMulKernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        K, M, N
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized tiled matrix multiplication (CUDA)");
}