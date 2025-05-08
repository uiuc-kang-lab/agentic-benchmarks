#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Store read-only threshold constants in constant memory for fast access
__constant__ float softplus_thresholds_f[2] = {20.0f, -20.0f};
__constant__ double softplus_thresholds_d[2] = {20.0, -20.0};

// Forward declaration of softplus compute function template
template <typename scalar_t>
__device__ __forceinline__ scalar_t softplus_compute(scalar_t x);

// Specialization for float
template <>
__device__ __forceinline__ float softplus_compute<float>(float x) {
    float threshold_hi = softplus_thresholds_f[0];
    float threshold_lo = softplus_thresholds_f[1];
    if (x > threshold_hi)
        return x;
    else if (x < threshold_lo)
        return expf(x);
    else
        float exp_neg_x = expf(-x); return (x > 0.0f) ? (x + log1pf(exp_neg_x)) : log1pf(expf(x));
}

// Specialization for double
template <>
__device__ __forceinline__ double softplus_compute<double>(double x) {
    double threshold_hi = softplus_thresholds_d[0];
    double threshold_lo = softplus_thresholds_d[1];
    if (x > threshold_hi)
        return x;
    else if (x < threshold_lo)
        return exp(x);
    else
        return (x > 0.0) ? (x + log1p(exp(-x))) : log1p(exp(x));
}

// CUDA kernel using grid-stride loop and __ldg for read-only access
template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < size; idx += stride) {
        scalar_t x = __ldg(&input[idx]);
        output[idx] = softplus_compute(x);
    }
}

// PyTorch CUDA forward function
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
