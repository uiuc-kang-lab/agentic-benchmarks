#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define VECTOR_SIZE 4  // Using float4 for vectorized loads/stores

__device__ __forceinline__ void load_vector_tile(const float* __restrict__ src,
                                                float dst[TILE_SIZE][TILE_SIZE+1],
                                                const int row, const int col,
                                                const int N, const bool transpose) {
    float4 vec;
    int base_idx = transpose ? (col * N + row) : (row * N + col);
    
    if (col + 4 <= N && row < N) {
        vec = *reinterpret_cast<const float4*>(&src[base_idx]);
        dst[threadIdx.y][threadIdx.x * 4] = vec.x;
        dst[threadIdx.y][threadIdx.x * 4 + 1] = vec.y;
        dst[threadIdx.y][threadIdx.x * 4 + 2] = vec.z;
        dst[threadIdx.y][threadIdx.x * 4 + 3] = vec.w;
    } else {
        for (int i = 0; i < 4; i++) {
            if (col + i < N && row < N) {
                dst[threadIdx.y][threadIdx.x * 4 + i] = src[base_idx + (transpose ? N : 1) * i];
            } else {
                dst[threadIdx.y][threadIdx.x * 4 + i] = 0.0f;
            }
        }
    }
}

__global__ void vectorized_triangular_mm_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * (TILE_SIZE/4) * 4 + threadIdx.x * 4;  // Vectorized column access

    if (row >= N || col >= N) return;

    // Handle upper triangular part
    if (row < col) {
        if (col + 4 <= N) {
            float4 zeros = {0.0f, 0.0f, 0.0f, 0.0f};
            *reinterpret_cast<float4*>(&C[row * N + col]) = zeros;
        } else {
            for (int i = 0; i < 4 && col + i < N; i++) {
                C[row * N + col + i] = 0.0f;
            }
        }
        return;
    }

    float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        const int tile_start = t * TILE_SIZE;
        if (tile_start > row) break;

        // Load tiles using vectorized loads where possible
        if (threadIdx.x < TILE_SIZE/4) {
            load_vector_tile(A, As, row, tile_start + threadIdx.x * 4, N, false);
            load_vector_tile(B, Bs, tile_start + threadIdx.y, col, N, true);
        }
        __syncthreads();

        const int k_start = max(tile_start, col);
        const int k_end = min(tile_start + TILE_SIZE, row + 1);

        #pragma unroll 4
        for (int k = k_start; k < k_end; k++) {
            const float a_val = As[threadIdx.y][k - tile_start];
            sum.x += a_val * Bs[k - tile_start][threadIdx.x * 4];
            sum.y += a_val * Bs[k - tile_start][threadIdx.x * 4 + 1];
            sum.z += a_val * Bs[k - tile_start][threadIdx.x * 4 + 2];
            sum.w += a_val * Bs[k - tile_start][threadIdx.x * 4 + 3];
        }
        __syncthreads();
    }

    // Store results using vectorized stores where possible
    if (col + 4 <= N) {
        *reinterpret_cast<float4*>(&C[row * N + col]) = sum;
    } else {
        for (int i = 0; i < 4 && col + i < N; i++) {
            float* sum_ptr = reinterpret_cast<float*>(&sum);
            C[row * N + col + i] = sum_ptr[i];
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

    dim3 block(TILE_SIZE/4, TILE_SIZE);  // Adjusted for vectorized access
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    vectorized_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized triangular matrix multiplication (CUDA)");
}