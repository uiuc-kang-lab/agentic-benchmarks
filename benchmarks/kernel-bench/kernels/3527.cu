#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float and double types
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

// CUDA kernel that applies the SELU activation ensuring memory coalescing.
// Each thread in a warp accesses consecutive memory locations, which maximizes
// memory bandwidth utilization on the NVIDIA H100 GPU.

template <typename scalar_t>
__global__ void selu_kernel_coalesced(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        size_t numel) {
    // Compute global index for each thread
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Use grid-stride loop so that each thread processes a contiguous chunk
    size_t stride = blockDim.x * gridDim.x;
    
    // Loop over elements in steps of 'stride'. When entering the loop,
    // threads in the same warp will access consecutive elements, ensuring
    // coalesced global memory accesses.
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t res = (x > static_cast<scalar_t>(0))
                           ? x
                           : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = static_cast<scalar_t>(1.05070098735548049342) * res;
    }
}

// Host function that sets up the kernel launch parameters and calls the kernel.

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    
    // Using 1024 threads per block for high occupancy
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_coalesced_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_coalesced<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Coalesced Memory Access (CUDA)");
}
