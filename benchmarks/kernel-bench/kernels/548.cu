#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__constant__ float d_scalar;

__global__ void multiplyKernel(const float* __restrict__ A,
                               float* __restrict__ C,
                               int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * d_scalar;
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    cudaMemcpyToSymbol(d_scalar, &s, sizeof(float));
    
    const int threads = 512;
    const int blocks = (size + threads - 1) / threads;

    multiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                       C.data_ptr<float>(),
                                       size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel");
}