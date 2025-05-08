#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Inline GELU function for float precision
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Combined kernel that processes the bulk of the data using vectorized loads with __ldg
// and handles remainder elements in a single launch. This reduces overall kernel launch overhead
// while ensuring coalesced memory accesses for the main aligned portion of the data.
__global__ void gelu_kernel_combined(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     size_t numel,
                                     size_t n4,
                                     size_t rem) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        // Process 4 elements at a time using vectorized loads (float4)
        const float4* input4 = reinterpret_cast<const float4*>(input);
        float4* output4 = reinterpret_cast<float4*>(output);
        float4 in_val = __ldg(&input4[idx]);
        in_val.x = gelu_function(in_val.x);
        in_val.y = gelu_function(in_val.y);
        in_val.z = gelu_function(in_val.z);
        in_val.w = gelu_function(in_val.w);
        output4[idx] = in_val;
    } else if (idx < n4 + rem) {
        // Process any leftover elements that don't form a complete float4
        size_t pos = idx - n4 + n4 * 4;  // Calculate the actual position in the array
        float val = __ldg(&input[pos]);
        output[pos] = gelu_function(val);
    }
}

// Forward function callable from Python
// This function computes the number of vectorized (float4) iterations and
// the remainder elements, then launches a single kernel to handle both cases.

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
                "Only float32 is supported for this optimized GELU version");

    auto output = torch::empty_like(x);
    const size_t numel = x.numel();
    const size_t vec_size = 4;  // Using float4 for vectorized access
    size_t n4 = numel / vec_size;
    size_t rem = numel % vec_size;

    // Total threads needed = number of vectorized iterations plus remainder elements
    size_t total_threads = n4 + rem;
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;

    gelu_kernel_combined<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        n4,
        rem);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with combined vectorized and scalar processing");
}
