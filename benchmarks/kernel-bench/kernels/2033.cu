#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define a macro for the maximum block size
#define MAX_BLOCK_SIZE 512

// Kernel function for lower triangular matrix multiplication with adaptive block size
__global__ void adaptive_block_size_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int N, int block_size) {
    extern __shared__ float shared_memory[];
    float* As = shared_memory;
    float* Bs = shared_memory + block_size * block_size;

    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;

    if (row >= N || col >= N) return;

    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    int num_tiles = (N + block_size - 1) / block_size;

    for (int t = 0; t < num_tiles; ++t) {
        int tile_start = t * block_size;
        if (tile_start > row) break;

        int a_col = tile_start + threadIdx.x;
        if (a_col < N) {
            As[threadIdx.y * block_size + threadIdx.x] = __ldg(&A[row * N + a_col]);
        } else {
            As[threadIdx.y * block_size + threadIdx.x] = 0.0f;
        }

        int b_row = tile_start + threadIdx.y;
        if (b_row < N) {
            Bs[threadIdx.y * block_size + threadIdx.x] = __ldg(&B[b_row * N + col]);
        } else {
            Bs[threadIdx.y * block_size + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        int k_start = max(tile_start, col);
        int k_end = min(tile_start + block_size, row + 1);

        for (int k = k_start; k < k_end; ++k) {
            int k_tile = k - tile_start;
            sum += As[threadIdx.y * block_size + k_tile] * Bs[k_tile * block_size + threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B, int block_size) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1) / block_size,
              (N + block_size - 1) / block_size);

    size_t shared_memory_size = 2 * block_size * block_size * sizeof(float);

    adaptive_block_size_kernel<<<grid, block, shared_memory_size>>>(A.data_ptr<float>(),
                                                                   B.data_ptr<float>(),
                                                                   C.data_ptr<float>(),
                                                                   N, block_size);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive block size lower triangular matrix multiplication (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("block_size"));
}