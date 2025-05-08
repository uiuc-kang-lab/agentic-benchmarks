#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute the Softplus activation
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(const scalar_t x) {
    if (x > static_cast<scalar_t>(20.0)) {
        return x;
    } else if (x < static_cast<scalar_t>(-20.0)) {
        return exp(x);
    }
    return log1p(exp(x));
}

// CUDA kernel with manual loop unrolling (factor of 4) using #pragma unroll
template <typename scalar_t>
__global__ void softplus_kernel_unrolled(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    int i = tid;
    // Unrolled loop: process 4 elements per iteration
    for (; i + 3 * stride < size; i += 4 * stride) {
        #pragma unroll
        {
            scalar_t in0 = input[i];
            scalar_t in1 = input[i + stride];
            scalar_t in2 = input[i + 2 * stride];
            scalar_t in3 = input[i + 3 * stride];
            
            output[i]             = compute_softplus(in0);
            output[i + stride]      = compute_softplus(in1);
            output[i + 2 * stride]  = compute_softplus(in2);
            output[i + 3 * stride]  = compute_softplus(in3);
        }
    }
    // Process any remaining elements
    for (; i < size; i += stride) {
        output[i] = compute_softplus(input[i]);
    }
}

// CUDA forward function
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_unrolled<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
