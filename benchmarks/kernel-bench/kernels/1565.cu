#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

__global__ void fused_pipelined_upper_triangular_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int tile_width
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        for (int k = row; k <= col; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor fused_pipelined_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);

    // Allocate the output tensor C
    auto C = torch::empty({N, N}, A.options());

    // Allocate device memory for A and B
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));
    cudaMemcpy(d_A, A.data_ptr<float>(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data_ptr<float>(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Use streams for pipelining
    cudaStream_t streams[2];
    cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernels alternating between streams
    for (int i = 0; i < 2; ++i) {
        fused_pipelined_upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock, 0, streams[i % 2]>>>(
            d_A, d_B, d_C, N, 256
        );
    }

    // Copy the result from the device to host
    cudaMemcpy(C.data_ptr<float>(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Synchronize
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    // Free device memory and destroy streams
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_pipelined_upper_triangular_matmul, "Fused pipelined upper triangular matrix multiplication");
}