#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Inline GELU function for float precision
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Hybrid optimized GELU kernel
__global__ void gelu_kernel_hybrid(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t numel) {
    extern __shared__ float shared_data[];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;

    // Use shared memory for loading
    if (idx < numel) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();

    // Process data using GELU
    if (idx < numel) {
        shared_data[tid] = gelu_function(shared_data[tid]);
    }
    __syncthreads();

    // Efficiently store back using shared memory
    if (idx < numel) {
        output[idx] = shared_data[tid];
    }
}

// Forward function callable from Python.
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
               "Only float32 is supported for this hybrid optimized version");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();
    
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    const size_t shared_mem_size = threads * sizeof(float);

    if (numel % 4 == 0) {
        // Use vectorized approach when elements are aligned
        gelu_kernel_vectorized_ldg<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            numel / 4);
    } else {
        // Use shared memory for non-aligned elements
        gelu_kernel_hybrid<<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            numel);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with hybrid optimization");
}