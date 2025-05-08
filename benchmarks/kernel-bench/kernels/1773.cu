#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to compute C = tril(A * B) for lower triangular matrices A and B.
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
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

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Asynchronously copy data to device
    cudaMemcpyAsync(A.data_ptr<float>(), A.data_ptr<float>(), A.numel() * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(B.data_ptr<float>(), B.data_ptr<float>(), B.numel() * sizeof(float), cudaMemcpyHostToDevice, stream2);

    // Launch the CUDA kernel in stream1
    triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, stream1>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    // Asynchronously copy result back to host
    cudaMemcpyAsync(C.data_ptr<float>(), C.data_ptr<float>(), C.numel() * sizeof(float), cudaMemcpyDeviceToHost, stream1);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication with streams (CUDA)");
}