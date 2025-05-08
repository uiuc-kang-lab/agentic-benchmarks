#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel that partitions the work evenly among blocks to ensure every thread gets a near-equal workload
// Each block processes a contiguous chunk of the input tensor
template <typename scalar_t>
__global__ void even_distrib_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                                 scalar_t* __restrict__ output,
                                                 size_t numel) {
    // Partition the total work among blocks
    size_t chunk = (numel + gridDim.x - 1) / gridDim.x;  // Compute the chunk size
    size_t start = chunk * blockIdx.x;                   // Start index for this block
    size_t end = start + chunk;                          
    if (end > numel) end = numel;                        

    // Each thread in the block processes its portion of the chunk using a stride loop
    for (size_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        scalar_t x = input[i];
        // Compute HardSigmoid: y = (x + 3) / 6, then clamp to [0,1]
        scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
        y = (y < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) : (y > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y);
        output[i] = y;
    }
}

// Host function to launch the kernel
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    // Choose a fixed number of threads per block and an appropriate number of blocks for even workload distribution
    const int threads = 1024;
    const int blocks = 256;  // Fixed grid size to partition the input evenly

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "even_distrib_hardsigmoid_cuda", ([&] {
        even_distrib_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA) with even block-level workload distribution");
}
