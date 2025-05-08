#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Store threshold constants in constant memory for both float and double types
__constant__ float c_upper_threshold_float = 20.0f;
__constant__ float c_lower_threshold_float = -20.0f;
__constant__ double c_upper_threshold_double = 20.0;
__constant__ double c_lower_threshold_double = -20.0;

// Optimized CUDA kernel with loop unrolling, constant memory, and optimal block size
template <typename scalar_t>
__global__ void optimized_softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Unroll loop by factor of 4
    for (; idx < size; idx += stride * 4) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int index = idx + i * stride;
            if (index < size) {
                scalar_t x = input[index];
                if constexpr (std::is_same<scalar_t, float>::value) {
                    if (x > c_upper_threshold_float) {
                        output[index] = x;
                    } else if (x < c_lower_threshold_float) {
                        output[index] = expf(x);
                    } else {
                        output[index] = log1pf(expf(x));
                    }
                } else {
                    if (x > c_upper_threshold_double) {
                        output[index] = x;
                    } else if (x < c_lower_threshold_double) {
                        output[index] = exp(x);
                    } else {
                        output[index] = log1p(exp(x));
                    }
                }
            }
        }
    }
}

// CUDA forward function
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 512; // Use optimal block size
    const int blocks = (size + threads * 4 - 1) / (threads * 4);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        optimized_softplus_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
