#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Inline GELU function using float precision
__device__ __forceinline__ float gelu_function(float x) {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Kernel that processes data in 128-bit chunks using float4 and __ldg for read-only loads
__global__ void gelu_kernel_vectorized_ldg(const float4* __restrict__ input,
                                            float4* __restrict__ output,
                                            size_t n4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        // Load using __ldg to utilize the read-only data cache
        float4 in_val = __ldg(&input[idx]);
        in_val.x = gelu_function(in_val.x);
        in_val.y = gelu_function(in_val.y);
        in_val.z = gelu_function(in_val.z);
        in_val.w = gelu_function(in_val.w);
        output[idx] = in_val;
    }
}

// Kernel to process remaining elements that are not divisible by 4
__global__ void gelu_kernel_remainder_ldg(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           size_t offset,
                                           size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pos = idx + offset;
    if (pos < numel) {
        float val = __ldg(&input[pos]);
        output[pos] = gelu_function(val);
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float, "Only float32 is supported");

    auto output = torch::empty_like(x);
    size_t numel = x.numel();
    const size_t vec_size = 4; // float4 processes 4 float elements at a time (128 bits)
    size_t n4 = numel / vec_size;
    size_t remainder = numel % vec_size;

    const int threads = 256;
    if (n4 > 0) {
        int blocks = (n4 + threads - 1) / threads;
        gelu_kernel_vectorized_ldg<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            n4);
    }
    if (remainder > 0) {
        int rem_blocks = (remainder + threads - 1) / threads;
        gelu_kernel_remainder_ldg<<<rem_blocks, threads>>>(
            x.data_ptr<float>() + n4 * vec_size,
            output.data_ptr<float>() + n4 * vec_size,
            n4 * vec_size,
            numel);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with __ldg and 128-bit aligned accesses");
}
