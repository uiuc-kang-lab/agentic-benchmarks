#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel_balanced(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size,
    const int elements_per_thread) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int start_idx = tid * elements_per_thread;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int idx = start_idx + i;
        if (idx < size) {
            const scalar_t val = input[idx];
            output[idx] = val > 0 ? val : 0;
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    // Optimize thread and block configuration
    const int threads_per_block = 128; // Multiple of warp size (32)
    const int elements_per_thread = 8;  // Each thread processes 8 elements
    const int total_threads_needed = (input.numel() + elements_per_thread - 1) / elements_per_thread;
    const int blocks = (total_threads_needed + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_balanced", ([&] {
        relu_kernel_balanced<scalar_t><<<blocks, threads_per_block>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel(),
            elements_per_thread
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward balanced (CUDA)");
}