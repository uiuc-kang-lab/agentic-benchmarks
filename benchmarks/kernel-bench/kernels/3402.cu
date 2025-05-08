#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Inline GELU function for float precision
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// CUDA kernel using shared memory with minimal synchronization
// Each thread loads one element into shared memory, __syncthreads() is called once to ensure data is loaded, 
// then each thread computes GELU and writes the result back to global memory. 
// This minimizes synchronization overhead by avoiding unnecessary __syncthreads() calls.
__global__ void gelu_kernel_shared_min_synch(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               size_t numel) {
    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load elements from global memory to shared memory if within bounds
    if (idx < numel) {
        s_data[tid] = input[idx];
    }
    // Synchronize only once to ensure all data for the block is loaded
    __syncthreads();

    // Compute GELU using data from shared memory and write back to global memory
    if (idx < numel) {
        float val = s_data[tid];
        val = gelu_function(val);
        output[idx] = val;
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
                "Only float32 is supported in this kernel");

    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    // Configure block and grid dimensions
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(float);

    gelu_kernel_shared_min_synch<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with minimal __syncthreads usage");
}
