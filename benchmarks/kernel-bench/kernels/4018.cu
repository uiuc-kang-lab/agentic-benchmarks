#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel combines vectorized float4 operations with shared memory tiling.
// Each thread loads a float4 element from global memory into shared memory,
// then computes the ELU activation, and writes the result back to global memory.
// A grid-stride loop is used to cover the entire input array of float4 elements.

__global__ void elu_kernel_vec4_shared(const float4* __restrict__ x, float4* __restrict__ out, float alpha, int n4) {
    extern __shared__ float4 tile[];  // shared memory tile for blockDim.x float4 elements
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop to process all float4 elements
    for (int idx = globalIdx; idx < n4; idx += stride) {
        // Load the float4 element from global memory into shared memory
        tile[threadIdx.x] = x[idx];
        __syncthreads();

        // Read from shared memory
        float4 val = tile[threadIdx.x];
        float4 result;
        result.x = (val.x > 0.0f) ? val.x : alpha * (expf(val.x) - 1.0f);
        result.y = (val.y > 0.0f) ? val.y : alpha * (expf(val.y) - 1.0f);
        result.z = (val.z > 0.0f) ? val.z : alpha * (expf(val.z) - 1.0f);
        result.w = (val.w > 0.0f) ? val.w : alpha * (expf(val.w) - 1.0f);
        __syncthreads();

        // Write the computed result back to global memory
        out[idx] = result;
        __syncthreads();
    }
}

// Interface function called from Python
// This function assumes that the number of elements in the input tensor is divisible by 4.
// Otherwise, a separate handling for remainder elements is needed.

torch::Tensor elu_cuda_vec4_shared(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    int n4 = n / 4;  // Number of float4 elements
    
    const int threads = 256;
    const int blocks = (n4 + threads - 1) / threads;
    
    // Allocate shared memory for a blockTile of float4 elements
    size_t sharedMemSize = threads * sizeof(float4);

    elu_kernel_vec4_shared<<<blocks, threads, sharedMemSize>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        alpha,
        n4
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_vec4_shared, "ELU activation vectorized with shared memory (CUDA)");
}
