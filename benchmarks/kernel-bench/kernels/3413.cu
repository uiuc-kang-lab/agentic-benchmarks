// Includes
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// GELU function for float
__device__ inline float gelu_function(float x) {
    // 1.4142135623730951 is sqrt(2)
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Kernel for the bulk of data using vectorized float4 loads/stores
__global__ void gelu_kernel_vectorized(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    size_t n4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 in_val = input[idx];
        // Apply GELU element-wise to each component
        in_val.x = gelu_function(in_val.x);
        in_val.y = gelu_function(in_val.y);
        in_val.z = gelu_function(in_val.z);
        in_val.w = gelu_function(in_val.w);
        output[idx] = in_val;
    }
}

// Kernel for processing the remainder elements using shared memory
__global__ void gelu_kernel_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t offset,     // Starting index for this kernel
    size_t numel) {    // Total number of elements

    extern __shared__ float s_data[];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    size_t global_idx = offset + idx;

    // Load remainder data into shared memory
    if (global_idx < numel) {
        s_data[tid] = input[global_idx];
    }
    __syncthreads();

    // Apply GELU in shared memory
    if (global_idx < numel) {
        s_data[tid] = gelu_function(s_data[tid]);
    }
    __syncthreads();

    // Write results back to global memory
    if (global_idx < numel) {
        output[global_idx] = s_data[tid];
    }
}

// Combined forward function callable from Python
// It uses vectorized accesses for the main chunk and shared memory for remainders

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
                "Only float32 is supported for this optimized version");

    auto output = torch::empty_like(x);
    const size_t numel = x.numel();
    const size_t vec_size = 4;  // using float4
    const size_t n4 = numel / vec_size;           // Number of vectorized groups
    const size_t remainder = numel % vec_size;      // Remaining elements

    const int threads = 256;

    // Launch vectorized kernel if there is a bulk of data
    if (n4 > 0) {
        int blocks = (n4 + threads - 1) / threads;
        gelu_kernel_vectorized<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            n4);
    }

    // Process remaining elements using the shared memory kernel
    if (remainder > 0) {
        int rem_blocks = (remainder + threads - 1) / threads;
        size_t shared_mem_size = threads * sizeof(float);
        // The offset is at n4 * vec_size
        gelu_kernel_shared<<<rem_blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "Optimized GELU activation forward (CUDA) using vectorization and shared memory");
}
