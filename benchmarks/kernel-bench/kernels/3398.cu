#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// GELU activation function
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Hybrid kernel combining vectorized loads with shared memory
__global__ void gelu_kernel_hybrid(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    size_t n4) {
    
    extern __shared__ float4 shared_data[];
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    
    // Vectorized load into shared memory
    if (idx < n4) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();
    
    // Process data in shared memory
    if (idx < n4) {
        float4 in4 = shared_data[tid];
        in4.x = gelu_function(in4.x);
        in4.y = gelu_function(in4.y);
        in4.z = gelu_function(in4.z);
        in4.w = gelu_function(in4.w);
        shared_data[tid] = in4;
    }
    __syncthreads();
    
    // Vectorized write back to global memory
    if (idx < n4) {
        output[idx] = shared_data[tid];
    }
}

// Handle remaining elements
__global__ void gelu_kernel_remainder(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t offset,
    size_t numel) {
    
    extern __shared__ float shared_data[];
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    if (idx + offset < numel) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();

    if (idx + offset < numel) {
        shared_data[tid] = gelu_function(shared_data[tid]);
    }
    __syncthreads();

    if (idx + offset < numel) {
        output[idx] = shared_data[tid];
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
               "Only float32 is supported");
    
    auto output = torch::empty_like(x);
    const size_t numel = x.numel();
    const size_t vec_size = 4;
    const size_t n4 = numel / vec_size;
    const size_t remainder = numel % vec_size;
    
    const int threads = 256;
    const int blocks = (n4 + threads - 1) / threads;
    const size_t shared_mem_size = threads * sizeof(float4);
    
    // Main vectorized kernel with shared memory
    gelu_kernel_hybrid<<<blocks, threads, shared_mem_size>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        n4);
    
    // Handle remaining elements using shared memory
    if (remainder > 0) {
        const int rem_blocks = (remainder + threads - 1) / threads;
        const size_t rem_shared_mem_size = threads * sizeof(float);
        gelu_kernel_remainder<<<rem_blocks, threads, rem_shared_mem_size>>>(
            x.data_ptr<float>() + n4 * vec_size,
            output.data_ptr<float>() + n4 * vec_size,
            n4 * vec_size,
            numel);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid GELU activation forward (CUDA)");
}