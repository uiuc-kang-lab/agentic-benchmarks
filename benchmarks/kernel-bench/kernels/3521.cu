#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: inline exponential function for float and double types
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

// CUDA kernel for SELU activation that avoids unnecessary atomic operations
// Each thread processes independent contiguous elements, eliminating race conditions
// and thus negating the need for atomic operations in global memory.

template <typename scalar_t>
__global__ void selu_kernel_atomic_optimized(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               size_t numel) {
    // Compute the unique index for the thread
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Loop over elements, each thread processing its own set of indices
    for (size_t i = idx; i < numel; i += stride) {
        // Load input using read-only cache
        scalar_t x = __ldg(&input[i]);
        // Compute SELU: if (x > 0) then x else alpha*(exp(x)-1)
        scalar_t result = (x > static_cast<scalar_t>(0)) 
                          ? x 
                          : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
        // Scale the result
        output[i] = static_cast<scalar_t>(1.05070098735548049342) * result;
    }
}

// Host function that launches the SELU activation kernel
// No atomic operations are used since each thread operates on separate elements.

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    const int threads = 1024;
    int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_atomic_optimized_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_atomic_optimized<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward (CUDA) - Atomic Optimized");
}
