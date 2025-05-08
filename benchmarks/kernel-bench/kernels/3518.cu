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

// Combined CUDA kernel that applies the SELU activation to each element.
// It uses loop unrolling for efficiency and balanced workload distribution.
template <typename scalar_t>
__global__ void selu_kernel_combined_optimized(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               size_t numel) {
    size_t total_threads = blockDim.x * gridDim.x;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t work_per_thread = (numel + total_threads - 1) / total_threads;
    size_t start = tid * work_per_thread;
    size_t end = start + work_per_thread;
    if (end > numel) end = numel;

    // Unroll the loop to process multiple elements per thread
    #pragma unroll 4
    for (size_t i = start; i < end; i++) {
        scalar_t x = __ldg(&input[i]);
        scalar_t result = (x > static_cast<scalar_t>(0))
                              ? x
                              : static_cast<scalar_t>(1.67326324235437728481) *
                                    (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = static_cast<scalar_t>(1.05070098735548049342) * result;
    }
}

// Host function that launches the combined optimized CUDA SELU kernel.
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_combined_optimized_cuda", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        scalar_t *output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_combined_optimized<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Combined Optimization (CUDA)");
}