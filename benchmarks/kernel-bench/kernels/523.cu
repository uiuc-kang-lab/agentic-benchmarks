#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernelCoalesced(const float* __restrict__ A,
                                         float* __restrict__ C,
                                         float s,
                                         int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride * 2) {
        int j = i + stride;
        C[i] = A[i] * s;
        if (j < size) {
            C[j] = A[j] * s;
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    multiplyKernelCoalesced<<<blocks, threads>>>(A.data_ptr<float>(),
                                                 C.data_ptr<float>(),
                                                 s,
                                                 size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced memory access matrix-scalar multiplication kernel");
}