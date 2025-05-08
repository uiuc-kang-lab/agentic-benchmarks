#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel that evenly partitions the workload among blocks and threads
template <typename scalar_t>
__global__ void evenload_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    // Determine the total number of blocks in the grid
    int num_blocks = gridDim.x;

    // Compute the base chunk size for each block and the remainder
    int chunk = size / num_blocks;
    int remainder = size % num_blocks;

    // Each block gets a contiguous subrange of the input
    // Blocks with index less than 'remainder' get one extra element
    int start = blockIdx.x * chunk + (blockIdx.x < remainder ? blockIdx.x : remainder);
    int end = start + chunk + (blockIdx.x < remainder ? 1 : 0);

    // Distribute the subrange work evenly among threads in the block
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        scalar_t val = input[i];
        output[i] = val > static_cast<scalar_t>(0) ? val : static_cast<scalar_t>(0);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    const int threads = 256;
    
    // Choose grid size. For small inputs, use one block per element; otherwise cap the number of blocks to a value (e.g., 1024).
    int blocks = (size < 1024) ? static_cast<int>(size) : 1024;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "evenload_relu_kernel", ([&] {
        evenload_relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Evenly load balanced ReLU forward (CUDA)");
}
