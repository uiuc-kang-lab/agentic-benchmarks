#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32

__device__ __forceinline__ void load_tile_coalesced(const float* src, float dst[TILE_SIZE][TILE_SIZE+1], 
                                                   int row, int tile_start, int N, bool is_a_matrix) {
    // Coalesced memory loading using vectorized loads where possible
    if (is_a_matrix) {
        // Load A matrix - row-major access
        float4* src_vec = (float4*)(&src[row * N + tile_start]);
        if (tile_start + threadIdx.x * 4 < N) {
            float4 tmp = __ldg(src_vec + threadIdx.x);
            dst[threadIdx.y][threadIdx.x * 4] = tmp.x;
            if (threadIdx.x * 4 + 1 < TILE_SIZE) dst[threadIdx.y][threadIdx.x * 4 + 1] = tmp.y;
            if (threadIdx.x * 4 + 2 < TILE_SIZE) dst[threadIdx.y][threadIdx.x * 4 + 2] = tmp.z;
            if (threadIdx.x * 4 + 3 < TILE_SIZE) dst[threadIdx.y][threadIdx.x * 4 + 3] = tmp.w;
        }
    } else {
        // Load B matrix - column-major access with padding to avoid bank conflicts
        if (tile_start + threadIdx.y < N) {
            dst[threadIdx.y][threadIdx.x] = __ldg(&src[(tile_start + threadIdx.y) * N + row]);
        } else {
            dst[threadIdx.y][threadIdx.x] = 0.0f;
        }
    }
}

__global__ void optimized_triangular_mm_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];

    // Block-wide indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;

    // Early exit for upper triangle
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Process tiles with vectorized loads and optimized memory access
    for (int t = 0; t < num_tiles; ++t) {
        int tile_start = t * TILE_SIZE;
        if (tile_start > row) break;

        // Coalesced loading of tiles
        load_tile_coalesced(A, As, row, tile_start, N, true);
        load_tile_coalesced(B, Bs, col, tile_start, N, false);
        
        __syncthreads();

        // Compute bounds for triangular multiplication
        int k_start = max(tile_start, col);
        int k_end = min(tile_start + TILE_SIZE, row + 1);

        // Use registers for temporary accumulation
        #pragma unroll 8
        for (int k = k_start; k < k_end; ++k) {
            int k_tile = k - tile_start;
            sum = fma(As[threadIdx.y][k_tile], Bs[k_tile][threadIdx.x], sum);
        }
        
        __syncthreads();
    }

    // Write result with coalesced access
    C[row * N + col] = sum;
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
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    optimized_triangular_mm_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                                   B.data_ptr<float>(),
                                                   C.data_ptr<float>(),
                                                   N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}