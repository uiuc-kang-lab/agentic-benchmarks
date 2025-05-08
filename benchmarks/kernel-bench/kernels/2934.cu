#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel that uses a stride loop to process all elements
// Each thread processes multiple elements spaced by the total number of threads

template <typename scalar_t>
__global__ void tanh_kernel_strided(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Stride loop to cover all elements even when size > total thread count
    for (; idx < size; idx += stride) {
        // Boundary check is implicit in the loop condition
        output[idx] = tanhf(input[idx]);
    }
}

// Forward function launching the CUDA kernel

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;  // Ensure full coverage

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_strided", ([&] {
        tanh_kernel_strided<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided Tanh forward (CUDA)");
}
