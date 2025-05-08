#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Device function for computing the Softplus activation in a numerically stable way
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(const scalar_t x) {
    if (x > static_cast<scalar_t>(20.0)) {
        return x;
    } else if (x < static_cast<scalar_t>(-20.0)) {
        return exp(x);
    }
    return log1p(exp(x));
}

// Kernel that evenly distributes the workload among all threads
// Each thread computes its contiguous chunk of the input array
template <typename scalar_t>
__global__ void softplus_kernel_even(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Compute the number of elements each thread should process (ceiling division)
    int chunk = (size + total_threads - 1) / total_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > size) end = size;

    for (int i = start; i < end; i++) {
        const scalar_t x = input[i];
        output[i] = compute_softplus(x);
    }
}

// CUDA forward function
// It calculates grid and block dimensions to evenly cover the input tensor
// and launches the kernel
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();

    // Launch configuration: choose 256 threads per block
    const int threads = 256;
    // Ensure we have enough threads to cover the workload evenly
    const int blocks = std::min(65535, (size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_even<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
