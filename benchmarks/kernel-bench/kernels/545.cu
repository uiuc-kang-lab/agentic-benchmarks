#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernel(const float* __restrict__ A,
                               float* __restrict__ C,
                               float s,
                               int64_t size)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x * 4 + tid;
    const unsigned int stride = blockDim.x;
    
    // Each thread processes 4 elements
    float4 elements;
    float4 result;
    
    if (gid + 3 * stride < size) {
        elements = *((float4*)&A[gid]);
        result.x = elements.x * s;
        result.y = elements.y * s;
        result.z = elements.z * s;
        result.w = elements.w * s;
        *((float4*)&C[gid]) = result;
    } else {
        // Handle remaining elements
        for (unsigned int i = 0; i < 4; i++) {
            const unsigned int curr_idx = gid + i * stride;
            if (curr_idx < size) {
                C[curr_idx] = A[curr_idx] * s;
            }
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
    const int blocks = (size + (threads * 4) - 1) / (threads * 4);

    multiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                       C.data_ptr<float>(),
                                       s,
                                       size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel");
}