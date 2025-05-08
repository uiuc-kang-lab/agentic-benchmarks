#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size (assumed to be a multiple of 4)
#define BLOCK_SIZE 32

// Optimized matrix multiplication kernel combining coalesced memory accesses,
// shared memory bank conflict avoidance (via padding), and manual loop unrolling
__global__ void matmul_opt_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    const int N) {
    // Use padded shared memory tiles to avoid bank conflicts
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float value = 0.0f;
    int tileCount = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < tileCount; t++) {
        int tiledCol = t * BLOCK_SIZE + tx;
        int tiledRow = t * BLOCK_SIZE + ty;

        // Load a tile of A into shared memory in a coalesced manner with bounds check
        if (row < N && tiledCol < N)
            s_A[ty][tx] = A[row * N + tiledCol];
        else
            s_A[ty][tx] = 0.0f;

        // Load a tile of B into shared memory in a coalesced manner with bounds check
        if (col < N && tiledRow < N)
            s_B[ty][tx] = B[tiledRow * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        // Compute the partial dot product with manual loop unrolling (factor of 4)
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += 4) {
            value = __fmaf_rn(s_A[ty][k],     s_B[k][tx],     value);
            value = __fmaf_rn(s_A[ty][k + 1], s_B[k + 1][tx], value);
            value = __fmaf_rn(s_A[ty][k + 2], s_B[k + 2][tx], value);
            value = __fmaf_rn(s_A[ty][k + 3], s_B[k + 3][tx], value);
        }

        __syncthreads();
    }

    // Write the computed result to global memory, using a bounds check
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// C++ interface for PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    const int N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_opt_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Matrix Multiplication (CUDA) combining coalesced access, bank conflict avoidance, and manual loop unrolling");
}
