#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// GELU function for float precision
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Vectorized kernel ensuring memory coalescing using float4
// Each thread loads a float4 (4 floats) from consecutive memory locations
// and applies the GELU function element-wise.
__global__ __launch_bounds__(256) void gelu_kernel_coalesced_vectorized(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    size_t n4) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    // Loop over input in steps of (blockDim.x * gridDim.x)
    for (; tid < n4; tid += stride) {
        // Use __ldg to load via the read-only cache
        float4 in_val = __ldg(&input[tid]);
        in_val.x = gelu_function(in_val.x);
        in_val.y = gelu_function(in_val.y);
        in_val.z = gelu_function(in_val.z);
        in_val.w = gelu_function(in_val.w);
        output[tid] = in_val;
    }
}

// Scalar kernel to process any remaining elements
__global__ void gelu_kernel_coalesced_scalar(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t offset,
    size_t numel) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (; tid + offset < numel; tid += stride) {
        float x = __ldg(&input[tid + offset]);
        output[tid + offset] = gelu_function(x);
    }
}

// Forward function callable from Python
// It splits the work into a vectorized path (using float4) and a scalar fallback for remainder elements.
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "Only float32 is supported");

    auto output = torch::empty_like(x);
    const size_t numel = x.numel();
    const size_t vec_size = 4; // Elements processed per float4
    const size_t n4 = numel / vec_size;
    const size_t remainder = numel % vec_size;

    int threads = 256;
    int blocks = (n4 + threads - 1) / threads;

    // Launch the vectorized kernel for aligned accesses
    if (n4 > 0) {
        gelu_kernel_coalesced_vectorized<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            n4);
    }

    // Process the remaining elements to ensure correctness
    if (remainder > 0) {
        int rem_blocks = (remainder + threads - 1) / threads;
        gelu_kernel_coalesced_scalar<<<rem_blocks, threads>>>(
            x.data_ptr<float>() + n4 * vec_size,
            output.data_ptr<float>() + n4 * vec_size,
            0,
            remainder);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with coalesced memory accesses");
}
