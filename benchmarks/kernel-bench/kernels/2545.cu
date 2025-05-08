#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation using strided loops
template <typename scalar_t>
__global__ void relu_kernel_strided(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread processes multiple elements, spaced by the grid-wide stride
    for (int i = idx; i < size; i += stride) {
        output[i] = input[i] > static_cast<scalar_t>(0) ? input[i] : static_cast<scalar_t>(0);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int threads = 1024; // Increased thread count per block
    const int blocks = (input.numel() + threads - 1) / threads; // Adjusting blocks calculation for stride

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_strided", ([&] {
        relu_kernel_strided<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward using strided loops (CUDA)");
}
