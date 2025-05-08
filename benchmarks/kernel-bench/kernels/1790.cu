#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Device function to load tile from matrix A into shared memory
__device__ void load_tile_A(float* As, const float* A, int row, int tile_idx, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (row < N && (tile_idx * TILE_SIZE + tx) <= row) {
        As[ty * TILE_SIZE + tx] = A[row * N + (tile_idx * TILE_SIZE + tx)];
    } else {
        As[ty * TILE_SIZE + tx] = 0.0f;
    }
}

// Device function to load tile from matrix B into shared memory
__device__ void load_tile_B(float* Bs, const float* B, int col, int tile_idx, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if ((tile_idx * TILE_SIZE + ty) < N && col < N) {
        Bs[ty * TILE_SIZE + tx] = B[(tile_idx * TILE_SIZE + ty) * N + col];
    } else {
        Bs[ty * TILE_SIZE + tx] = 0.0f;
    }
}

// Device function to compute partial sum for a tile
__device__ float compute_tile_sum(const float* As, const float* Bs, 
                                int row, int col, int tile_idx) {
    float sum = 0.0f;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    #pragma unroll 8
    for (int k = 0; k < TILE_SIZE; k++) {
        if ((tile_idx * TILE_SIZE + k) >= col && (tile_idx * TILE_SIZE + k) <= row) {
            sum += As[ty * TILE_SIZE + k] * Bs[k * TILE_SIZE + tx];
        }
    }
    return sum;
}

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[TILE_SIZE * TILE_SIZE];
    __shared__ float Bs[TILE_SIZE * TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.0f;
            return;
        }
        
        // Loop over tiles
        for (int t = col/TILE_SIZE; t <= row/TILE_SIZE; t++) {
            // Load tiles collaboratively
            load_tile_A(As, A, row, t, N);
            load_tile_B(Bs, B, col, t, N);
            
            __syncthreads();
            
            // Compute partial sum for this tile
            if (row >= col) {
                sum += compute_tile_sum(As, Bs, row, col, t);
            }
            
            __syncthreads();
        }
        
        if (row >= col) {
            C[row * N + col] = sum;
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular triangular matrix multiplication (CUDA)");
}