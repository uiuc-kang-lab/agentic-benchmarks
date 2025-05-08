#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Explicit specializations of gelu_function for float
template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

__global__ void gelu_kernel_combined(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Vectorized processing with float4 for memory coalescing and efficient DRAM access
    for (size_t i = idx * 4; i < numel; i += stride * 4) {
        if (i + 3 < numel) {
            float4 in4 = reinterpret_cast<const float4*>(x)[i / 4];
            in4.x = gelu_function(in4.x);
            in4.y = gelu_function(in4.y);
            in4.z = gelu_function(in4.z);
            in4.w = gelu_function(in4.w);
            reinterpret_cast<float4*>(y)[i / 4] = in4;
        }
    }

    // Process any remaining elements individually
    for (size_t i = idx * 4 + 4 * (numel / 4); i < numel; i += stride) {
        if (i < numel) {
            float val = __ldg(&x[i]);
            y[i] = gelu_function(val);
        }
    }
}

torch::Tensor forward_combined(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
               "Only float32 is supported for combined version");

    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    int threads = 1024;
    int blocks = (numel + threads * 4 - 1) / (threads * 4); // dividing by 4 for vectorized float4 processing

    gelu_kernel_combined<<<blocks, threads>>>(x.data_ptr<float>(),
                                              output.data_ptr<float>(),
                                              numel);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_combined", &forward_combined, "GELU activation forward (CUDA) combined efficient version");
}