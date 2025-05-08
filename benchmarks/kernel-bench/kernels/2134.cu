#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to compute C = tril(A * B) for lower triangular matrices A and B.
__global__ void optimized_triangular_mm_kernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                int N) {
    extern __shared__ float shared_data[];
    float* shared_A = shared_data;
    float* shared_B = shared_data + blockDim.x * blockDim.y;

    // Compute the row and column indices for this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i <= row; i += blockDim.x) {
        if (row < N && i + threadIdx.x <= row) {
            shared_A[threadIdx.y * blockDim.x + threadIdx.x] = 
                A[row * N + i + threadIdx.x];
        }
        if (col < N && i + threadIdx.y <= col) {
            shared_B[threadIdx.y * blockDim.x + threadIdx.x] = 
                B[(i + threadIdx.y) * N + col];
        }

        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k) {
            if (i + k <= row && i + k <= col) {
                sum += shared_A[threadIdx.y * blockDim.x + k] * 
                       shared_B[k * blockDim.x + threadIdx.x];
            }
        }

        __syncthreads();
    }

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

    size_t shared_memory_size = 2 * threads * threads * sizeof(float);

    // Launch the CUDA kernel.
    optimized_triangular_mm_kernel<<<numBlocks, threadsPerBlock, shared_memory_size>>>(
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
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}