#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void triangular_mm_coalesced(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (row >= N || col >= N) return;
    
    if (row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.f;

    // Calculate starting and ending tiles based on valid k range [col, row]
    int m_start = col / TILE_SIZE;
    int m_end = row / TILE_SIZE;
    
    for (int m = m_start; m <= m_end; ++m) {
        int tile_base = m * TILE_SIZE;
        int tile_end = min(tile_base + TILE_SIZE, N);

        // Load A tile with optimized coalescing
        int load_col_a = tile_base + threadIdx.x;
        if (row < N && load_col_a < N) {
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + load_col_a]);
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.f;
        }

        // Load B tile with transpose for coalesced access
        int load_row_b = tile_base + threadIdx.x;
        if (load_row_b < N && col < N) {
            sB[threadIdx.x][threadIdx.y] = __ldg(&B[load_row_b * N + col]);
        } else {
            sB[threadIdx.x][threadIdx.y] = 0.f;
        }
        __syncthreads();

        // Calculate valid k range within current tile
        int local_start = max(col - tile_base, 0);
        int local_end = min(row - tile_base + 1, TILE_SIZE);

        // Accumulate valid elements
        for (int k = local_start; k < local_end; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.y]; // Access transposed B
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrix squaring required");
    int N = A.size(0);

    auto C = torch::zeros_like(A);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_coalesced<<<grid, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Triangular MM (CUDA)");
}
