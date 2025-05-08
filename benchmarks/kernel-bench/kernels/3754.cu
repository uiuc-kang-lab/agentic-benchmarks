#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with block-stride loop and __ldg for read-only cache to improve memory coalescing

template <typename scalar_t>
__global__ void softplus_kernel_coalesced(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (; idx < size; idx += stride) {
        // Use __ldg to load from global memory via read-only cache
        scalar_t x = __ldg(&input[idx]);

        if (x > static_cast<scalar_t>(20.0)) {
            output[idx] = x;
        } else if (x < static_cast<scalar_t>(-20.0)) {
            output[idx] = exp(x);
        } else {
            output[idx] = log1p(exp(x));
        }
    }
}


// CUDA forward function

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_coalesced<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
