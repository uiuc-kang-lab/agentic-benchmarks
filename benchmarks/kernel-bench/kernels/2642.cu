#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation with balanced workload
template <typename scalar_t>
__global__ void relu_kernel_balanced(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    // Distribute workload evenly across threads
    for (int i = idx; i < size; i += total_threads) {
        scalar_t val = input[i];
        output[i] = val > 0 ? val : 0;
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = std::min(65535, int((input.numel() + threads - 1) / threads));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_balanced", ([&] {
        relu_kernel_balanced<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with balanced workload (CUDA)");
}