#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Inline GELU function for float precision
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Vectorized CUDA kernel minimizing warp divergence
__global__ void gelu_kernel_min_warp_div(const float4* __restrict__ input,
                                          float4* __restrict__ output,
                                          size_t n4, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n4; i += stride) {
        float4 in_val = input[i];
        in_val.x = gelu_function(in_val.x);
        in_val.y = gelu_function(in_val.y);
        in_val.z = gelu_function(in_val.z);
        in_val.w = gelu_function(in_val.w);
        output[i] = in_val;
    }

    // Handle remaining elements within the same loop to avoid branching
    size_t remainder_start = n4 * 4;
    for (size_t i = idx + remainder_start; i < numel; i += stride) {
        float val = reinterpret_cast<const float*>(input)[i];
        output[i] = gelu_function(val);
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
                "Only float32 is supported for the warp divergence reduced version");

    auto output = torch::empty_like(x);
    size_t numel = x.numel();
    const size_t vec_size = 4; // float4 => 4 floats (128 bits)
    size_t n4 = numel / vec_size;

    int threads = 256;
    int blocks = (n4 + threads - 1) / threads;

    // Launch the kernel minimizing warp divergence
    gelu_kernel_min_warp_div<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        n4, numel);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with minimized warp divergence");
}
