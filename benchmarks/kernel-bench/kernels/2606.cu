#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel using a grid-stride loop and __ldg for read caching
// Combines a robust iteration pattern with simple element-wise ReLU logic

template <typename scalar_t>
__global__ void grid_stride_relu_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t size) {

    // Shared memory tile
    extern __shared__ scalar_t shared_mem[];
    
    const int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    const int stride = gridDim.x * blockDim.x;
    const int TILE_SIZE = blockDim.x;

    // Grid-stride loop with tiling
    for (; idx < size; idx += stride) {
        // Load tile into shared memory
        if (idx < size) {
            shared_mem[tid] = __ldg(&input[idx]);
        }
        __syncthreads();

        // Process data in shared memory
        if (idx < size) {
            scalar_t x = shared_mem[tid];
            output[idx] = (x > static_cast<scalar_t>(0)) ? x : static_cast<scalar_t>(0);
        }
        __syncthreads();
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_stride_relu_kernel", ([&] {
        grid_stride_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward using grid-stride loop (CUDA)");
}
