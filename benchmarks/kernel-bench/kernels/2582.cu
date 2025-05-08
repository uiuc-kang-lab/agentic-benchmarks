#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel_2d(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    // 2D grid for better occupancy
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;
    const int bid = blockIdx.x + blockIdx.y * gridDim.x;
    const int idx = tid + bid * block_size;
    
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    // Use 16x16 thread blocks
    dim3 threads(16, 16);
    const int total_threads = threads.x * threads.y;
    
    // Calculate grid dimensions to precisely cover the elements
    const int64_t num_elements = input.numel();
    const int blocks_x = (num_elements + total_threads - 1) / total_threads;
    // Use a reasonable number of blocks in y-dimension for better load distribution
    const int blocks_y = (blocks_x + 31) / 32;  // Limit y-dimension, adjust x accordingly
    const int final_blocks_x = (blocks_x + blocks_y - 1) / blocks_y;
    dim3 blocks(final_blocks_x, blocks_y);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_2d", ([&] {
        relu_kernel_2d<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward (CUDA)");
}