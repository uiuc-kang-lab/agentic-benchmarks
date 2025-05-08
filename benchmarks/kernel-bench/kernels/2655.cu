#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Store the constant threshold (0) in constant memory
__constant__ float c_threshold_f = 0.0f;
__constant__ double c_threshold_d = 0.0;

// Helper template to fetch the constant threshold value for each supported type
template <typename scalar_t>
__device__ inline scalar_t get_constant_threshold();

template <>
__device__ inline float get_constant_threshold<float>() {
    return c_threshold_f;
}

template <>
__device__ inline double get_constant_threshold<double>() {
    return c_threshold_d;
}

// CUDA kernel for ReLU activation using constant memory for the threshold
template <typename scalar_t>
__global__ void relu_kernel_constant(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t threshold = get_constant_threshold<scalar_t>();  // fetch constant threshold
        scalar_t val = input[idx];
        output[idx] = (val > threshold) ? val : static_cast<scalar_t>(0);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 128;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_constant", ([&] {
        relu_kernel_constant<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with constant memory threshold (CUDA)");
}
