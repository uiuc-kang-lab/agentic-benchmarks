#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(scalar_t x);

template<>
__device__ __forceinline__ float compute_softplus<float>(float x) {
    const float THRESH_HI = 20.0f;
    const float THRESH_LO = -20.0f;
    if (x > THRESH_HI) return x;
    if (x < THRESH_LO) return __expf(x);
    return __log1pf(__expf(x));
}

template<>
__device__ __forceinline__ double compute_softplus<double>(double x) {
    const double THRESH_HI = 20.0;
    const double THRESH_LO = -20.0;
    if (x > THRESH_HI) return x;
    if (x < THRESH_LO) return exp(x);
    return log1p(exp(x));
}

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t x = __ldg(&input[idx]);
        output[idx] = compute_softplus(x);
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

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