#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to compute C = tril(A * B) for lower triangular matrices A and B.
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int N) {
    extern __shared__ float sharedMemory[];
    float* s_A = sharedMemory;
    float* s_B = sharedMemory + blockDim.x;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.x;

    float sum = 0.0f;

    if (row < N && col < N) {
        for (int k = 0; k <= row; k += blockDim.x) {
            if (threadId + k <= row) {
                s_A[threadId] = A[row * N + (threadId + k)];
                s_B[threadId] = B[(threadId + k) * N + col];
            }
            __syncthreads();

            for (int j = 0; j < blockDim.x && (j + k <= row); ++j) {
                sum += s_A[j] * s_B[j];
            }
            __syncthreads();
        }

        if (row >= col) {
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

    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);
    size_t sharedMemSize = 2 * threads * sizeof(float);

    // Launch the CUDA kernel
    triangular_mm_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication with shared memory (CUDA)");
}