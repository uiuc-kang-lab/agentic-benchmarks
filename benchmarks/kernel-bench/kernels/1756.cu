#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 32;

typedef float4 aligned_float;

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int N) {
    __shared__ aligned_float A_tile[TILE_SIZE][TILE_SIZE/4];
    __shared__ aligned_float B_tile[TILE_SIZE][TILE_SIZE/4];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    if (row < N && col < N && row >= col) {
        for (int k_tile = blockIdx.x; k_tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++k_tile) {
            int tiled_row = k_tile * TILE_SIZE + threadIdx.y;
            int tiled_col = k_tile * TILE_SIZE + threadIdx.x;

            // Aligned 128-bit loads with __ldg for A
            if (row < N && tiled_col < N && tiled_col <= row) {
                const float4* ptr = reinterpret_cast<const float4*>(&A[row * N + tiled_col]);
                A_tile[threadIdx.y][threadIdx.x/4] = __ldg(ptr);
            } else {
                A_tile[threadIdx.y][threadIdx.x/4] = {0,0,0,0};
            }

            // Aligned 128-bit loads with __ldg for B
            if (tiled_row < N && col < N && col <= tiled_row) {
                const float4* ptr = reinterpret_cast<const float4*>(&B[tiled_row * N + col]);
                B_tile[threadIdx.y][threadIdx.x/4] = __ldg(ptr);
            } else {
                B_tile[threadIdx.y][threadIdx.x/4] = {0,0,0,0};
            }

            __syncthreads();

            for (int k = 0; k < TILE_SIZE; ++k) {
                int global_k = k_tile * TILE_SIZE + k;
                if (global_k >= col && global_k <= row) {
                    sum += A_tile[threadIdx.y][k/4].x * B_tile[k][threadIdx.x/4].x;
                    sum += A_tile[threadIdx.y][k/4].y * B_tile[k][threadIdx.x/4].y;
                    sum += A_tile[threadIdx.y][k/4].z * B_tile[k][threadIdx.x/4].z;
                    sum += A_tile[threadIdx.y][k/4].w * B_tile[k][threadIdx.x/4].w;
                }
            }
            __syncthreads();
        }
        C[row * N + col] = sum;
    } else if (row < N && col < N) {
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<blocks, threads>>>( 
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
    m.def("forward", &forward, "Triangular MM with LDG and aligned accesses (CUDA)");
}