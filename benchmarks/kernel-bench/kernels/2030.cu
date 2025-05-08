#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__device__ __forceinline__ void load_a_tile_coalesced(const float* A, int row, int tile_start, int N, float As[TILE_SIZE][TILE_SIZE+1]) {
    for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
        int a_col = tile_start + i;
        if (a_col < N) {
            As[threadIdx.y][i] = __ldg(&A[row * N + a_col]);
        } else {
            As[threadIdx.y][i] = 0.0f;
        }
    }
}

__device__ __forceinline__ void load_b_tile_coalesced(const float* B, int col, int tile_start, int N, float Bs[TILE_SIZE][TILE_SIZE+1]) {
    for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
        int b_row = tile_start + i;
        if (b_row < N) {
            Bs[i][threadIdx.x] = __ldg(&B[b_row * N + col]);
        } else {
            Bs[i][threadIdx.x] = 0.0f;
        }
    }
}

__global__ void aligned_coalesced_triangular_mm_kernel(const float* __restrict__ A,
                                                        const float* __restrict__ B,
                                                        float* __restrict__ C,
                                                        int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (row >= N || col >= N) return;

    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int tile_start = t * TILE_SIZE;
        if (tile_start > row) break;

        load_a_tile_coalesced(A, row, tile_start, N, As);
        load_b_tile_coalesced(B, col, tile_start, N, Bs);
        __syncthreads();

        int k_start = max(tile_start, col);
        int k_end = min(tile_start + TILE_SIZE, row + 1);

        for (int k = k_start; k < k_end; ++k) {
            int k_tile = k - tile_start;
            sum += As[threadIdx.y][k_tile] * Bs[k_tile][threadIdx.x];
        }
        __syncthreads();
    }

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

    aligned_coalesced_triangular_mm_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                                            B.data_ptr<float>(),
                                                            C.data_ptr<float>(),
                                                            N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Aligned and Coalesced Triangular Matrix Multiplication (CUDA)");
}
