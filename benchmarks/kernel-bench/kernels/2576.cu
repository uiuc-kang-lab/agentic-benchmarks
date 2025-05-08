#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory to load a tile of input data, apply the ReLU activation, and write back to global memory.
// Note: __syncthreads() is NOT used because each thread only operates on its own element, avoiding unnecessary synchronization overhead.

template <typename scalar_t>
__global__ void tiled_shared_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    // Allocate shared memory dynamically
    extern __shared__ scalar_t shared_data[];

    // Compute global index for each thread
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load one element per thread from global memory into shared memory if within bounds
    if (global_idx < size) {
        shared_data[threadIdx.x] = input[global_idx];
    }
    // No __syncthreads() is used here because each thread only relies on its own element and no inter-thread dependency exists.

    // Process the loaded value and write it back to global memory
    if (global_idx < size) {
        scalar_t val = shared_data[threadIdx.x];
        output[global_idx] = val > static_cast<scalar_t>(0) ? val : static_cast<scalar_t>(0);
    }
}

// PyTorch wrapper function

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    // Allocate shared memory: one element per thread
    size_t shared_mem_size = threads * input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tiled_shared_relu_kernel", ([&] {
        tiled_shared_relu_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled Shared Memory ReLU forward (CUDA) with minimal __syncthreads()");
}
