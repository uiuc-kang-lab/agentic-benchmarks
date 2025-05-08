#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N
) {
    constexpr int TILE_SIZE = 32;
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Iterate over tiles
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile into shared memory
        const int tile_row = threadIdx.y;
        const int tile_col = threadIdx.x;
        const int tile_idx = t * TILE_SIZE;

        if ((row < N) && (tile_idx + tile_col < N)) {
            s_A[tile_row][tile_col] = A[row * N + tile_idx + tile_col];
        } else {
            s_A[tile_row][tile_col] = 0.0f;
        }

        if ((tile_idx + tile_row < N) && (col < N)) {
            s_B[tile_row][tile_col] = B[(tile_idx + tile_row) * N + col];
        } else {
            s_B[tile_row][tile_col] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum for this tile
        if (row < N && col < N && row >= col) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                const int global_k = tile_idx + k;
                if (global_k >= col && global_k <= row) {
                    sum += s_A[tile_row][k] * s_B[k][tile_col];
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

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1) && A.size(0) == B.size(0),
                "Matrices must be square and of the same size");

    const int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    constexpr int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_shared_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication with shared memory (CUDA)");
}