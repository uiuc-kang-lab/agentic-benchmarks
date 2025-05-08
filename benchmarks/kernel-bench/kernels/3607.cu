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

// CUDA kernel that applies the SELU activation to each element.
// Note: Atomic operations are not used because each thread writes to a unique element,
// thereby avoiding race conditions and global memory contention.

template <typename scalar_t>
__global__ void selu_kernel_no_atomic(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);

    // Grid-stride loop to cover all elements
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t result = (x > static_cast<scalar_t>(0))
                              ? x
                              : alpha * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = lambda * result;
    }
}

// Host function that launches the SELU kernel without any atomic operations.
// Since each thread processes a unique element, race conditions are inherently avoided.
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_no_atomic", ([&] {
        selu_kernel_no_atomic<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward without Atomic Operations (CUDA)");
}
