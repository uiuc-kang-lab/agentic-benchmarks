#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel that leverages shared memory to load a block's worth of data,
// compute the HardSigmoid activation on it, and store the results back to global memory.
// HardSigmoid: y = clamp((x + 3) / 6, 0, 1)

template <typename scalar_t>
__global__ void hardsigmoid_shared_kernel(const scalar_t* __restrict__ input,
                                           scalar_t* __restrict__ output,
                                           size_t numel) {
    // Dynamically-allocated shared memory buffer
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Process the input in a grid-stride loop
    for (size_t idx = global_idx; idx < numel; idx += stride) {
        // Load one element from global memory into shared memory
        sdata[tid] = input[idx];
        __syncthreads();

        // Compute HardSigmoid using data in shared memory
        scalar_t x = sdata[tid];
        scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
        y = (y < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) 
             : (y > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1) : y;

        // Write the result back to global memory
        output[idx] = y;
        __syncthreads();
    }
}


torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    // Allocate shared memory: one element per thread
    size_t shared_memory_size = threads * input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_shared_optimized_cuda", ([&] {
        hardsigmoid_shared_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward with shared memory (CUDA)");
}
