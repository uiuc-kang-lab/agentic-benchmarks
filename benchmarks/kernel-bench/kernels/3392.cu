#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Explicit specializations of gelu_function for float
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Optimized CUDA kernel that applies the GELU activation element-wise
// using shared memory to reduce global memory accesses
__global__ void gelu_kernel_shared_memory(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t numel) {
    extern __shared__ float shared_data[];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < numel) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();

    // Apply GELU function
    if (idx < numel) {
        shared_data[tid] = gelu_function(shared_data[tid]);
    }
    __syncthreads();

    // Write back to global memory
    if (idx < numel) {
        output[idx] = shared_data[tid];
    }
}

// Forward function callable from Python.
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
               "Only float32 is supported for this optimized version");
    
    auto output = torch::empty_like(x);
    const size_t numel = x.numel();
    
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    const size_t shared_mem_size = threads * sizeof(float);
    
    gelu_kernel_shared_memory<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with shared memory optimization");
}