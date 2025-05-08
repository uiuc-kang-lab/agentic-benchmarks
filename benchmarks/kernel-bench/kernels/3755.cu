#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Restructuring kernel with 2D block and thread indexing for better mapping to 1D problem domains

template <typename scalar_t>
__global__ void softplus_kernel_2d(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int idx = block_id * (blockDim.x * blockDim.y) + thread_id;

    if (idx < size) {
        const scalar_t x = input[idx];
        if (x > static_cast<scalar_t>(20.0)) {
            output[idx] = x;
        } else if (x < static_cast<scalar_t>(-20.0)) {
            output[idx] = exp(x);
        } else {
            output[idx] = log1p(exp(x));
        }
    }
}


torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads_x = 16;
    const int threads_y = 16;
    const dim3 threads(threads_x, threads_y);
    const int blocks_x = (size + threads_x * threads_y - 1) / (threads_x * threads_y);
    const int blocks_y = 1;
    const dim3 blocks(blocks_x, blocks_y);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_2d<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
