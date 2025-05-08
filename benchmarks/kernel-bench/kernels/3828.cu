#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int unroll_factor = 4;
    
    // Process 4 elements per thread
    for (int idx = tid; idx < size; idx += stride * unroll_factor) {
        #pragma unroll
        for (int i = 0; i < unroll_factor; i++) {
            if (idx + i * stride < size) {
                const scalar_t x = input[idx + i * stride];
                if (x > 20.0) {
                    output[idx + i * stride] = x;
                } else if (x < -20.0) {
                    output[idx + i * stride] = exp(x);
                } else {
                    output[idx + i * stride] = log1p(exp(x));
                }
            }
        }
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads * 4 - 1) / (threads * 4);

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