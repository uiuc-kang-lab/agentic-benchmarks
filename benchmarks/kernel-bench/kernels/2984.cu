#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int TILE_WIDTH = 32;

template <typename scalar_t>
__global__ void tanh_kernel_shared_memory(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    __shared__ scalar_t tile[TILE_WIDTH];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    if (idx < size) {
        // Load data into shared memory block tile
        tile[tid] = input[idx];
        __syncthreads();

        // Apply tanh on shared memory and store in the output
        output[idx] = tanhf(tile[tid]);
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = TILE_WIDTH;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "tanh_kernel_shared_memory", ([&] {
        tanh_kernel_shared_memory<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with shared memory (CUDA)");
}
