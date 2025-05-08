#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void triangular_mm_kernel_shared(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            const int N) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t <= row/TILE_DIM; t++) {
        // Load tile from A and B into shared memory
        if (row < N && (t * TILE_DIM + threadIdx.x) <= row) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_DIM + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((t * TILE_DIM + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        if (row < N && col < N && row >= col) {
            for (int k = 0; k < TILE_DIM; k++) {
                if ((t * TILE_DIM + k) <= row && (t * TILE_DIM + k) >= col) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        if (row >= col) {
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    triangular_mm_kernel_shared<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}