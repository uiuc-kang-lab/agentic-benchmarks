#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel for Softplus
// Distributes workloads evenly across threads and blocks

template <typename scalar_t>
__global__ void softplus_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Loop over elements with stride
    for (int i = idx; i < size; i += stride) {
        const scalar_t x = input[i];
        if (x > 20.0) {
            output[i] = x;
        } else if (x < -20.0) {
            output[i] = exp(x);
        } else {
            output[i] = log1p(exp(x));
        }
    }
}

torch::Tensor softplus_cuda_forward_optimized(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda_optimized", ([&] {
        softplus_kernel_optimized<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_optimized", &softplus_cuda_forward_optimized, "Softplus forward optimized (CUDA)");
}