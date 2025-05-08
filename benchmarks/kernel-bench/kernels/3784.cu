#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define threshold values in constant memory
__constant__ float UPPER_THRESHOLD = 20.0f;
__constant__ float LOWER_THRESHOLD = -20.0f;
__constant__ double UPPER_THRESHOLD_DOUBLE = 20.0;
__constant__ double LOWER_THRESHOLD_DOUBLE = -20.0;

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(const scalar_t x) {
    if constexpr (std::is_same<scalar_t, float>::value) {
        if (x > UPPER_THRESHOLD) {
            return x;
        } else if (x < LOWER_THRESHOLD) {
            return expf(x);
        }
        return log1pf(expf(x));
    } else {
        if (x > UPPER_THRESHOLD_DOUBLE) {
            return x;
        } else if (x < LOWER_THRESHOLD_DOUBLE) {
            return exp(x);
        }
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
    const int threads = 512;
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