#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size,
    const int offset) {
    
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_idx = gid + offset;
    
    if (global_idx < size) {
        const scalar_t x = input[global_idx];
        scalar_t val;
        
        if (x > 20.0) {
            val = x;
        } else if (x < -20.0) {
            val = exp(x);
        } else {
            val = log1p(exp(x));
        }
        
        output[global_idx] = val;
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
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