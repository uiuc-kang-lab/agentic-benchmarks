#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation ensuring memory coalescing
template <typename scalar_t>
__global__ void relu_kernel_coalesced(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_size = 32;
    int idx = ((tid / warp_size) * warp_size * 4) + (tid % warp_size);
    
    // Process four consecutive elements per warp to ensure coalesced access
    for (int i = 0; i < 4 && idx + i < size; ++i) {
        const scalar_t val = input[idx + i * warp_size];
        output[idx + i * warp_size] = val > 0 ? val : 0;
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_coalesced", ([&] {
        relu_kernel_coalesced<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with memory coalescing (CUDA)");
}