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

// CUDA kernel that applies the SELU activation to each element with memory coalescing.
template <typename scalar_t>
__global__ void selu_kernel_coalesced(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);

    size_t i = idx;
    // Unroll factor: process four elements per iteration for improved performance
    while (i + 3 * stride < numel) {
        scalar_t x0 = input[i];
        scalar_t x1 = input[i + stride];
        scalar_t x2 = input[i + 2 * stride];
        scalar_t x3 = input[i + 3 * stride];

        output[i] = lambda * ((x0 > static_cast<scalar_t>(0))
                              ? x0
                              : alpha * (my_exp(x0) - static_cast<scalar_t>(1)));
        output[i + stride] = lambda * ((x1 > static_cast<scalar_t>(0))
                              ? x1
                              : alpha * (my_exp(x1) - static_cast<scalar_t>(1)));
        output[i + 2 * stride] = lambda * ((x2 > static_cast<scalar_t>(0))
                              ? x2
                              : alpha * (my_exp(x2) - static_cast<scalar_t>(1)));
        output[i + 3 * stride] = lambda * ((x3 > static_cast<scalar_t>(0))
                              ? x3
                              : alpha * (my_exp(x3) - static_cast<scalar_t>(1)));

        i += stride * 4;
    }
    // Process any remaining elements
    for (; i < numel; i += stride) {
        scalar_t x = input[i];
        output[i] = lambda * ((x > static_cast<scalar_t>(0))
                              ? x
                              : alpha * (my_exp(x) - static_cast<scalar_t>(1)));
    }
}

// Host function that launches the CUDA SELU kernel with memory coalescing using streams.
torch::Tensor selu_forward_stream(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda_stream", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        scalar_t *output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_coalesced<scalar_t><<<blocks, threads, 0, stream>>>(input_ptr, output_ptr, numel);
    }));

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward_stream, "SELU Activation Forward Coalesced with Streams (CUDA)");
}