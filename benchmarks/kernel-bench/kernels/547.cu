#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernelOptimized(const float* __restrict__ A,
                                         float* __restrict__ C,
                                         float s,
                                         int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 512;  // Increased number of threads per block
    const int warpSize = 32; // Size of a warp
    const int alignedThreads = (threads + warpSize - 1) / warpSize * warpSize; // Align threads to warp size
    const int blocks = (size + threads - 1) / threads;

    multiplyKernelOptimized<<<blocks, threads>>>(A.data_ptr<float>(),
                                                 C.data_ptr<float>(),
                                                 s,
                                                 size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Matrix-scalar multiplication kernel");
}