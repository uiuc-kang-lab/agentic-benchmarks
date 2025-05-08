#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void matmul_coalesced_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       const int N) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 to avoid bank conflicts
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 to avoid bank conflicts

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_SIZE;
    const int by = blockIdx.y * BLOCK_SIZE;

    // Global memory indices for coalesced access
    const int row = by + ty;
    const int col = bx + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        const int tileIdx = t * BLOCK_SIZE;
        
        // Coalesced load of A tile - threads in a warp read consecutive elements
        if (row < N && (tileIdx + tx) < N) {
            s_A[ty][tx] = A[row * N + (tileIdx + tx)];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Coalesced load of B tile - threads in a warp read consecutive elements
        if (col < N && (tileIdx + ty) < N) {
            s_B[ty][tx] = B[(tileIdx + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product with unrolled loop
        #pragma unroll 8
        for (int k = 0; k < BLOCK_SIZE; k += 4) {
            sum = __fmaf_rn(s_A[ty][k],   s_B[k][tx],   sum);
            sum = __fmaf_rn(s_A[ty][k+1], s_B[k+1][tx], sum);
            sum = __fmaf_rn(s_A[ty][k+2], s_B[k+2][tx], sum);
            sum = __fmaf_rn(s_A[ty][k+3], s_B[k+3][tx], sum);
        }

        __syncthreads();
    }

    // Coalesced write to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

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

    // Launch configuration
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_coalesced_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Matrix Multiplication (CUDA)");
}