#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to compute C = tril(A * B) for lower triangular matrices A and B with optimization.
__global__ void optimized_triang_mm_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int N) {
    // Shared memory for caching tiles of matrices A and B.
    __shared__ float tile_A[16][16];
    __shared__ float tile_B[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles of the input matrices.
    for (int m = 0; m < (N + 15) / 16; ++m) {
        // Load tiles into shared memory.
        if (row < N && (m * 16 + threadIdx.x) <= row) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + m * 16 + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (m * 16 + threadIdx.y) <= col) {
            tile_B[threadIdx.y][threadIdx.x] = B[(m * 16 + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute matrix multiplication using the tiles.
        for (int e = 0; e < 16; ++e) {
            sum += tile_A[threadIdx.y][e] * tile_B[e][threadIdx.x];
        }

        __syncthreads();
    }

    // Store the result in the lower triangular part of C.
    if (row < N && col < N && row >= col) {
        C[row * N + col] = sum;
    }
}

// C++ interface exposed to PyTorch.
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

    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    // Launch the CUDA kernel.
    optimized_triang_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication optimized (CUDA)");
}
