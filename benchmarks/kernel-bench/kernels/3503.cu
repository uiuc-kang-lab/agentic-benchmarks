#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu_function(scalar_t x) {
    const scalar_t sqrt_2_inv = 0.7071067811865475;
    return x * 0.5 * (1.0 + erf(x * sqrt_2_inv));
}

template <>
__device__ __forceinline__ float gelu_function<float>(float x) {
    const float sqrt_2_inv = 0.7071067811865475f;
    return x * 0.5f * (1.0f + erff(x * sqrt_2_inv));
}

template <typename scalar_t>
__global__ void gelu_kernel(const scalar_t* __restrict__ input,
                            scalar_t* __restrict__ output,
                            size_t n) {
    constexpr int vec_size = (sizeof(scalar_t) == 4) ? 4 : 2;
    using vec_t = typename std::conditional<vec_size == 4, float4, double2>::type;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t vec_offset = idx * vec_size;

    // Process aligned vector loads
    if (vec_offset + vec_size <= n) {
        vec_t in = *reinterpret_cast<const vec_t*>(input + vec_offset);
        vec_t out;
        if constexpr (vec_size == 4) {
            out.x = gelu_function(in.x);
            out.y = gelu_function(in.y);
            out.z = gelu_function(in.z);
            out.w = gelu_function(in.w);
        } else {
            out.x = gelu_function(in.x);
            out.y = gelu_function(in.y);
        }
        *reinterpret_cast<vec_t*>(output + vec_offset) = out;
    }
    // Handle remainder elements
    else if (vec_offset < n) {
        for (int i = 0; i < vec_size && vec_offset + i < n; i++) {
            output[vec_offset + i] = gelu_function(__ldg(input + vec_offset + i));
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    auto y = torch::empty_like(x);
    size_t n = x.numel();

    constexpr int threads = 256;
    int blocks = (n + threads * 4 - 1) / (threads * 4);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_forward", [&] {
        gelu_kernel<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                 y.data_ptr<scalar_t>(), n);
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU forward with vector loads (CUDA)");
}
