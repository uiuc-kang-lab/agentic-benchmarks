#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute softplus value in a device function.
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(const scalar_t x) {
    if (x > static_cast<scalar_t>(20.0)) {
        return x;
    } else if (x < static_cast<scalar_t>(-20.0)) {
        return exp(x);
    }
    return log1p(exp(x));
}

// Kernel that uses shared memory to load global input and then computes softplus.
// __syncthreads() is called only once after loading to ensure shared memory consistency.

template <typename scalar_t>
__global__ void softplus_kernel_shared(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    extern __shared__ scalar_t shared_input[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data from global memory into shared memory if within bounds.
    if (idx < size) {
        shared_input[threadIdx.x] = input[idx];
    }
    
    // Synchronize threads to ensure the entire block has loaded its data.
    __syncthreads();
    
    // Use the value from shared memory to compute softplus and store back to global memory.
    if (idx < size) {
        output[idx] = compute_softplus(shared_input[threadIdx.x]);
    }
}

// CUDA forward function that launches the kernel with dynamic shared memory size.

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda_shared", ([&] {
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
