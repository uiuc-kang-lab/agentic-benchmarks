#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// GELU function for float precision
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Shared memory kernel with minimal synchronization
// Loads a tile from global memory, synchronizes once to ensure all data is loaded,
// then performs the GELU computation and writes the result back to global memory.
__global__ void gelu_kernel_shared_min_sync(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             size_t numel) {
    extern __shared__ float tile[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads its own element into shared memory if within bounds
    if (idx < numel) {
        tile[threadIdx.x] = input[idx];
    }

    // Synchronize to ensure complete tile load; necessary for consistency
    __syncthreads();

    // Each thread reads its value from shared memory and computes GELU
    if (idx < numel) {
        float val = tile[threadIdx.x];
        output[idx] = gelu_function(val);
    }
}

// Forward function callable from PyTorch
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float, "Only float32 is supported");

    auto output = torch::empty_like(x);
    size_t numel = x.numel();

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(float);

    gelu_kernel_shared_min_sync<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with minimal synchronization");
}
