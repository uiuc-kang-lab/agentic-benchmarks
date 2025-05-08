#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 32  // Warp-size dimensions for shuffle

__global__ void warp_reduce_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= N || col >= N || row < col) return;
    
    float sum = 0.0f;
    const int lane_id = threadIdx.x % 32;
    
    // Process 4 elements at a time with vectorized loads
    for (int k = col; k <= row; k += 4) {
        float4 a_chunk = *reinterpret_cast<const float4*>(&A[row * N + k]);
        for (int i = 0; i < 4 && (k+i) <= row; ++i) {
            float b_val = B[(k+i) * N + col];
            sum += a_chunk.x * b_val;
            a_chunk = __builtin_shuffle(a_chunk, 0x39);  // Rotate elements right
        }
    }
    
    // Warp-level reduction for sum aggregation
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write final result from lane 0 of each warp
    if (lane_id == 0) {
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    warp_reduce_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error:", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized triangular matmul (CUDA)");
}