#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to compute C = tril(A * B) for lower triangular matrices A and B using shared memory.
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    // Shared memory for tiles of A and B
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    // Compute the row and column indices for this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.f;

    // Loop over tiles
    for (int tileIdx = 0; tileIdx < (N + 31) / 32; ++tileIdx) {
        // Load tiles into shared memory
        if (row < N && tileIdx * 32 + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + tileIdx * 32 + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.f;

        if (tileIdx * 32 + threadIdx.y < N && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(tileIdx * 32 + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < 32; ++k) {
            if (row >= tileIdx * 32 + k && col <= tileIdx * 32 + k) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Write result
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

    // Define block and grid dimensions.
    const int threads = 32;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    // Launch the CUDA kernel.
    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}