#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x) {
    const scalar_t sqrt_2_inv = 1 / sqrt(2.0);
    return x * 0.5 * (1.0 + erf(x * sqrt_2_inv));
}

template <typename scalar_t, int VEC_SIZE>
__global__ void gelu_coalesced_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      size_t numel) {
    const size_t stride = blockDim.x * gridDim.x * VEC_SIZE;
    size_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;

    while (base_idx < numel) {
        if (base_idx + VEC_SIZE <= numel) {
            // Full vector processing (aligned access)
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                scalar_t val = input[base_idx + i];
                output[base_idx + i] = gelu_function(val);
            }
        } else {
            // Handle partial vector at end
            for (int i = 0; i < VEC_SIZE; ++i) {
                if (base_idx + i < numel) {
                    scalar_t val = input[base_idx + i];
                    output[base_idx + i] = gelu_function(val);
                }
            }
        }
        base_idx += stride;
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    auto y = torch::empty_like(x);
    const size_t n = x.numel();
    
    constexpr int VEC_SIZE = 4;
    const int threads = 256;
    const int blocks = std::min(static_cast<int>((n + (threads * VEC_SIZE) - 1) / (threads * VEC_SIZE)), 128);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_forward", [&] {
        gelu_coalesced_kernel<scalar_t, VEC_SIZE>
            <<<blocks, threads>>>(x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU forward with coalesced memory access");
}