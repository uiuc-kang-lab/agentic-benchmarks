#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to compute C = tril(A * B) for lower triangular matrices A and B using shared memory.
__global__ void triangular_mm_kernel_shared(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N) {
    // Compute the row and column indices for this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for tiles of A and B
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    float sum = 0.f;

    // Loop over tiles in the row and column
    for (int t = 0; t < (N + 15) / 16; ++t) {
        // Load tiles into shared memory
        if (row < N && t * 16 + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * 16 + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.f;

        if (col < N && t * 16 + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < 16; ++k) {
            if (t * 16 + k <= row && t * 16 + k <= col) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Write the result to the output matrix
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            C[row * N + col] = sum;
        }
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

    // Define block and grid dimensions.
    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    // Launch the CUDA kernel.
    triangular_mm_kernel_shared<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    // Check for kernel launch errors.
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication with shared memory (CUDA)");
}