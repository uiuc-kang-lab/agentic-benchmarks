#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernel(const float* __restrict__ A,
                               float* __restrict__ C,
                               float s,
                               int64_t size)
{
    // Ensure 128-bit aligned access by using vectorized loads
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;
    int vecSize = size / 4;  // Number of float4 segments
    int tail = size % 4;
    
    // Process vectorized segments
    for (int v = tid; v < vecSize; v += totalThreads) {
        float4 a = __ldg(reinterpret_cast<const float4*>(A) + v);
        a.x *= s;
        a.y *= s;
        a.z *= s;
        a.w *= s;
        reinterpret_cast<float4*>(C)[v] = a;
    }
    
    // Process tail elements
    int base = vecSize * 4;
    for (int i = tid; i < tail; i += totalThreads) {
        C[base + i] = __ldg(A + base + i) * s;
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 128;  // Changed block size to 128 threads
    const int blocks = (size + threads * 4 - 1) / (threads * 4);

    multiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                       C.data_ptr<float>(),
                                       s,
                                       size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel");
}