#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define WARP_SIZE 32

__global__ void warp_optimized_upper_triangular_kernel(const float* __restrict__ A,
                                                      const float* __restrict__ B,
                                                      float* __restrict__ C,
                                                      int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        for (int k = row; k <= col; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        // Warp-level reduction
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // The first thread in the warp writes the result
        if (lane == 0) {
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor warp_optimized_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(WARP_SIZE, 8);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    warp_optimized_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_optimized_upper_triangular_matmul, "Warp-optimized upper triangular matrix multiplication");
}