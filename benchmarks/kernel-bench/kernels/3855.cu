#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ float softplus_constants[2] = {20.0f, -20.0f};

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(scalar_t x) {
    if (x > softplus_constants[0]) {
        return x;
    } else if (x < softplus_constants[1]) {
        return exp(x);
    } else {
        return log1p(exp(x));
    }
}

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const scalar_t x = input[idx];
        output[idx] = compute_softplus(x);
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads; if (blocks > 65535) blocks = 65535;

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