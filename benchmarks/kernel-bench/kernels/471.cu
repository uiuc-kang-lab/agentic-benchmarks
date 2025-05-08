#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void stridedMultiplyKernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       float s,
                                       int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (idx < size) {
        C[idx] = A[idx] * s;
        idx += stride;
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

    stridedMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                               C.data_ptr<float>(),
                                               s,
                                               size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided matrix-scalar multiplication kernel");
}