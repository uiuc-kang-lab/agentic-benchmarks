#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define ALIGN_MASK (~3)  // For 128-bit alignment (4 floats)

__global__ void aligned_ldg_triangular_mm_kernel(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  const int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Early exit for out-of-bounds threads
    if (row >= N || col >= N) return;

    // Quick exit for upper triangular region blocks
    if (blockIdx.y * TILE_SIZE + TILE_SIZE - 1 < blockIdx.x * TILE_SIZE) {
        C[row * N + col] = 0.0f;
        return;
    }

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;

    // Calculate aligned boundaries for vectorized loads
    int aligned_col = col & ALIGN_MASK;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int m = 0; m < num_tiles; m++) {
        int tile_start = m * TILE_SIZE;
        
        // Load data into shared memory using vectorized loads where possible
        if ((threadIdx.x & 3) == 0 && tile_start + threadIdx.x + 3 < N) {
            // Vector load for aligned addresses
            float4 vecA = *reinterpret_cast<const float4*>(&A[row * N + tile_start + threadIdx.x]);
            sA[threadIdx.y][threadIdx.x] = __ldg(&vecA.x);
            sA[threadIdx.y][threadIdx.x + 1] = __ldg(&vecA.y);
            sA[threadIdx.y][threadIdx.x + 2] = __ldg(&vecA.z);
            sA[threadIdx.y][threadIdx.x + 3] = __ldg(&vecA.w);
        } else if (tile_start + threadIdx.x < N) {
            // Regular load for unaligned addresses
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + tile_start + threadIdx.x]);
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((threadIdx.y & 3) == 0 && tile_start + threadIdx.y + 3 < N) {
            // Vector load for aligned addresses
            float4 vecB = *reinterpret_cast<const float4*>(&B[(tile_start + threadIdx.y) * N + col]);
            sB[threadIdx.y][threadIdx.x] = __ldg(&vecB.x);
            sB[threadIdx.y + 1][threadIdx.x] = __ldg(&vecB.y);
            sB[threadIdx.y + 2][threadIdx.x] = __ldg(&vecB.z);
            sB[threadIdx.y + 3][threadIdx.x] = __ldg(&vecB.w);
        } else if (tile_start + threadIdx.y < N) {
            // Regular load for unaligned addresses
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[(tile_start + threadIdx.y) * N + col]);
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute valid range for this tile
        int k_start = max(col, tile_start);
        int k_end = min(row + 1, min(tile_start + TILE_SIZE, N));
        
        // Convert to local indices
        int local_start = k_start - tile_start;
        int local_end = k_end - tile_start;

        // Compute partial sum for this tile
        #pragma unroll 8
        for (int k = local_start; k < local_end; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row >= col) {
        C[row * N + col] = sum;
    } else {
        C[row * N + col] = 0.0f;
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
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    aligned_ldg_triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Aligned LDG Triangular Matrix Multiplication (CUDA)");
}