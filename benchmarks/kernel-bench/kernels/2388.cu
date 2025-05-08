#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size and coarsening factor
const int BLOCK_SIZE = 16;
const int COARSENING = 2; // Each thread computes two output columns

// Kernel that computes two output elements (columns) per thread with manual loop unrolling
__global__ void even_matmul_transposed_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int M, int N, int K) {
    // Compute row in the output matrix
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    // Compute base column index for this block
    int col_base = blockIdx.x * (BLOCK_SIZE * COARSENING);
    // Each thread computes two columns: col0 and col1
    int col0 = col_base + threadIdx.x;
    int col1 = col_base + threadIdx.x + BLOCK_SIZE;  // second output column

    // Accumulators for the two output elements
    float c0 = 0.0f;
    float c1 = 0.0f;

    // Shared memory tiles: one for A and two for B corresponding to each column group
    __shared__ float As[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs0[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs1[2][BLOCK_SIZE][BLOCK_SIZE];

    // Loop over tiles in K dimension
    for (int k_offset = 0; k_offset < K; k_offset += BLOCK_SIZE) {
        // Load tile for A
        if (row < M && (k_offset + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + k_offset + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile for B corresponding to col0 (B used in transposed manner)
        if (col0 < N && (k_offset + threadIdx.y) < K) {
            Bs0[threadIdx.y][threadIdx.x] = B[col0 * K + k_offset + threadIdx.y];
        } else {
            Bs0[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Load tile for B corresponding to col1
        if (col1 < N && (k_offset + threadIdx.y) < K) {
            Bs1[threadIdx.y][threadIdx.x] = B[col1 * K + k_offset + threadIdx.y];
        } else {
            Bs1[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Manually unrolled inner loop: compute partial dot-products
        // Unroll factor of 4 assuming BLOCK_SIZE is a multiple of 4 (16 in our case)
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += 4) {
            c0 = fmaf(As[threadIdx.y][k],     Bs0[k][threadIdx.x], c0);
            c0 = fmaf(As[threadIdx.y][k + 1], Bs0[k + 1][threadIdx.x], c0);
            c0 = fmaf(As[threadIdx.y][k + 2], Bs0[k + 2][threadIdx.x], c0);
            c0 = fmaf(As[threadIdx.y][k + 3], Bs0[k + 3][threadIdx.x], c0);

            c1 = fmaf(As[threadIdx.y][k],     Bs1[k][threadIdx.x], c1);
            c1 = fmaf(As[threadIdx.y][k + 1], Bs1[k + 1][threadIdx.x], c1);
            c1 = fmaf(As[threadIdx.y][k + 2], Bs1[k + 2][threadIdx.x], c1);
            c1 = fmaf(As[threadIdx.y][k + 3], Bs1[k + 3][threadIdx.x], c1);
        }

        __syncthreads();
    }

    // Write the computed results back to C if in bounds
    if (row < M) {
        if (col0 < N) {
            C[row * N + col0] = c0;
        }
        if (col1 < N) {
            C[row * N + col1] = c1;
        }
    }
}

// Forward function called from PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());
    
    // Grid dimensions: each block computes a tile of size (BLOCK_SIZE, BLOCK_SIZE * COARSENING)
    dim3 grid((N + BLOCK_SIZE * COARSENING - 1) / (BLOCK_SIZE * COARSENING), (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    
    even_matmul_transposed_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using manual loop unrolling (CUDA)");
}
