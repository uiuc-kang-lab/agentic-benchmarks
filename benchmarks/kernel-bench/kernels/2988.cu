#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_kernel_2d(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;
    const int bid = blockIdx.x + blockIdx.y * gridDim.x;
    const int idx = bid * block_size + tid;
    
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const dim3 threads(32, 16);  // 32x16 = 512 threads per block
    const int total_threads = threads.x * threads.y;
    const int num_blocks = (input.numel() + total_threads - 1) / total_threads;
    
    const int grid_y = (int)sqrt((float)num_blocks);
    const int grid_x = (num_blocks + grid_y - 1) / grid_y;
    const dim3 blocks(grid_x, grid_y);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "tanh_kernel_2d", ([&] {
        tanh_kernel_2d<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with 2D blocks (CUDA)");
}