#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float and double.
template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) {
    return expf(x);
}

template <>
__device__ inline double my_exp<double>(double x) {
    return exp(x);
}

// Optimized CUDA kernel that distributes work evenly and applies SELU activation.
template <typename scalar_t>
__global__ void selu_kernel_optimized(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = blockDim.x * gridDim.x;
    size_t work_per_thread = (numel + total_threads - 1) / total_threads;
    size_t start = idx * work_per_thread;
    size_t end = start + work_per_thread;
    if (end > numel) end = numel;

    for (size_t i = start; i < end; i++) {
        scalar_t x = input[i];
        scalar_t y = (x > static_cast<scalar_t>(0))
                         ? x
                         : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = static_cast<scalar_t>(1.05070098735548049342) * y;
    }
}

// Host function that launches the optimized CUDA SELU kernel.
torch::Tensor selu_forward_optimized(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_optimized_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_optimized<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward_optimized, "SELU Activation Forward (CUDA) with Optimized Workload Distribution");
}