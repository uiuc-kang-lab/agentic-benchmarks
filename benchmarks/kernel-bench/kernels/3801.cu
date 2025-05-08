#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// GPU kernel to compute the softplus function in a numerically stable manner
// using a grid-stride loop and dynamic block size selection based on input size.

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    while (idx < size) {
        scalar_t x = input[idx];
        // Use thresholds cast to the same type as x
        if (x > static_cast<scalar_t>(20.0)) {
            output[idx] = x;
        } else if (x < static_cast<scalar_t>(-20.0)) {
            output[idx] = exp(x);
        } else {
            // Use a numerically stable formulation for softplus
            // if x > 0 : softplus(x) = x + log1p(exp(-x)); else : log1p(exp(x))
            if (x > static_cast<scalar_t>(0.0))
                output[idx] = x + log1p(exp(-x));
            else
                output[idx] = log1p(exp(x));
        }
        idx += stride;
    }
}

// Forward function: dynamically selects the block size based on input size to experiment with different configurations

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();

    // Dynamically choose threads per block based on the total number of elements.
    int threads;
    if (size < 1024) {
        threads = 32;
    } else if (size < 8192) {
        threads = 64;
    } else if (size < 65536) {
        threads = 128;
    } else if (size < 262144) {
        threads = 256;
    } else {
        threads = 512;
    }
    const int blocks = (size + threads - 1) / threads; if (blocks > 65535) blocks = 65535;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
