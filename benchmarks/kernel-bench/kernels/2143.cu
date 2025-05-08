#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= N || col >= N) return;
    
    if (row < col) {
        C[row * N + col] = 0.f;
    } else {
        float sum = 0.f;
        // Use __ldg() for read-only loads and aligned memory access
        for (int k = col; k <= row; ++k) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }
        // Align write to 128-bit boundary using float4 store
        if ((row * N + col) % 4 == 0) {
            *reinterpret_cast<float4*>(&C[row * N + col]) = make_float4(sum, 0.f, 0.f, 0.f);
        } else {
            C[row * N + col] = sum;
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(16, 16);
    dim3 blocks((N + 15)/16, (N + 15)/16);
    
    triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matmul");
}