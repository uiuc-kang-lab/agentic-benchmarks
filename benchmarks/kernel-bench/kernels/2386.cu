#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size and coarsening factor
const int BLOCK_SIZE = 16;
const int COARSENING = 2; // Each thread computes two output columns

// Kernel that computes two output elements per thread with manual loop unrolling in the inner loop
__global__ void even_matmul_transposed_kernel_unroll(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Compute row index for A and C
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    // Compute base column index for C in this block
    int col_base = blockIdx.x * (BLOCK_SIZE * COARSENING);
    // Each thread computes two columns: col0 and col1
    int col0 = col_base + threadIdx.x;
    int col1 = col_base + threadIdx.x + BLOCK_SIZE;  // second column in this block tile

    // Accumulators for the two output elements
    float c0 = 0.0f;
    float c1 = 0.0f;

    // Shared memory tiles for A and B (B accessed in transposed manner)
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs0[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs1[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over tiles in the K dimension
    for (int k_offset = 0; k_offset < K; k_offset += BLOCK_SIZE) {
        // Load A tile
        if (row < M && (k_offset + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + k_offset + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tiles for both output groups (B is used in transposed fashion)
        if (col0 < N && (k_offset + threadIdx.y) < K) {
            Bs0[threadIdx.y][threadIdx.x] = B[col0 * K + k_offset + threadIdx.y];
        } else {
            Bs0[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (col1 < N && (k_offset + threadIdx.y) < K) {
            Bs1[threadIdx.y][threadIdx.x] = B[col1 * K + k_offset + threadIdx.y];
        } else {
            Bs1[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Manually unroll the inner loop over the tile dimension by a factor of 4
        // BLOCK_SIZE is assumed to be divisible by 4 (16 in our case)
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += 4) {
            float a0 = As[threadIdx.y][k];
            float a1 = As[threadIdx.y][k+1];
            float a2 = As[threadIdx.y][k+2];
            float a3 = As[threadIdx.y][k+3];

            float b0_0 = Bs0[k][threadIdx.x];
            float b0_1 = Bs0[k+1][threadIdx.x];
            float b0_2 = Bs0[k+2][threadIdx.x];
            float b0_3 = Bs0[k+3][threadIdx.x];

            float b1_0 = Bs1[k][threadIdx.x];
            float b1_1 = Bs1[k+1][threadIdx.x];
            float b1_2 = Bs1[k+2][threadIdx.x];
            float b1_3 = Bs1[k+3][threadIdx.x];

            c0 = fmaf(a0, b0_0, c0);
            c0 = fmaf(a1, b0_1, c0);
            c0 = fmaf(a2, b0_2, c0);
            c0 = fmaf(a3, b0_3, c0);

            c1 = fmaf(a0, b1_0, c1);
            c1 = fmaf(a1, b1_1, c1);
            c1 = fmaf(a2, b1_2, c1);
            c1 = fmaf(a3, b1_3, c1);
        }
        __syncthreads();
    }

    // Write the computed results back to global memory if within bounds
    if (row < M) {
        if (col0 < N)
            C[row * N + col0] = c0;
        if (col1 < N)
            C[row * N + col1] = c1;
    }
}

// Forward function exposed to PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same inner dimension K");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Grid dimensions: each block computes a tile of size (BLOCK_SIZE, BLOCK_SIZE * COARSENING)
    dim3 grid((N + BLOCK_SIZE * COARSENING - 1) / (BLOCK_SIZE * COARSENING), (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    even_matmul_transposed_kernel_unroll<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix multiplication with transposed B using manual loop unrolling (CUDA)");
}
