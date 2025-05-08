#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for computing the Softplus activation
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(const scalar_t x) {
    if (x > static_cast<scalar_t>(20.0)) {
        return x;
    } else if (x < static_cast<scalar_t>(-20.0)) {
        return exp(x);
    } else {
        return log1p(exp(x));
    }
}

// CUDA kernel that leverages shared memory to reduce global memory latency by loading a tile of input data
// into shared memory, then computing the Softplus activation and writing back the result.

template <typename scalar_t>
__global__ void softplus_kernel_shared(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         const int size) {
    // Allocate shared memory dynamically
    extern __shared__ scalar_t s_data[];

    // Compute global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input element into shared memory if within bounds
    if (idx < size) {
        s_data[threadIdx.x] = input[idx];
    }
    __syncthreads();

    // Compute softplus using the data in shared memory and write result to global memory
    if (idx < size) {
        scalar_t x = s_data[threadIdx.x];
        output[idx] = compute_softplus(x);
    }
}

// CUDA forward function that configures and launches the kernel

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 512;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_shared<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
