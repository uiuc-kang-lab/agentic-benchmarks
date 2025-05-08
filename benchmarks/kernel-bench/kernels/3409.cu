#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Modular device function to compute the GELU activation for a single float
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Modular device function to process a float4 vector using __ldg for read-only cache
__device__ inline float4 process_float4(const float4* input, int idx) {
    // Load a vector of 4 floats
    float4 val = __ldg(&input[idx]);
    // Apply GELU activation element-wise
    val.x = gelu_function(val.x);
    val.y = gelu_function(val.y);
    val.z = gelu_function(val.z);
    val.w = gelu_function(val.w);
    return val;
}

// Kernel for vectorized processing using float4
__global__ void vectorized_gelu_kernel(const float4* __restrict__ input,
                                         float4* __restrict__ output,
                                         size_t n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        output[idx] = process_float4(input, idx);
    }
}

// Modular device function to process a single float element
__device__ inline float process_scalar(const float* input, int idx) {
    float val = __ldg(&input[idx]);
    return gelu_function(val);
}

// Kernel for scalar processing of the remaining elements
__global__ void scalar_gelu_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      size_t offset,
                                      size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pos = idx + offset;
    if (pos < numel) {
        output[pos] = process_scalar(input, pos);
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float, "Only float32 is supported");

    auto output = torch::empty_like(x);
    size_t numel = x.numel();
    const size_t vec_size = 4; // Number of floats in a float4
    size_t n4 = numel / vec_size;
    size_t remainder = numel % vec_size;

    int threads = 256;
    int blocks = (n4 + threads - 1) / threads;

    // Launch the vectorized kernel for the bulk of the data
    if (n4 > 0) {
        vectorized_gelu_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            n4);
    }

    // Launch a scalar kernel to process any remaining elements
    if (remainder > 0) {
        int scalar_blocks = (remainder + threads - 1) / threads;
        scalar_gelu_kernel<<<scalar_blocks, threads>>>(
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
    m.def("forward", &forward, "GELU activation forward (CUDA) with modular device functions");
}
