#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized compute_softplus to be used inside the kernel
__device__ __forceinline__ float compute_softplus_device(const float x) {
    // Directly incorporate numeric checks within the device function for efficiency
    if (x > 20.0f) {
        return x;
    } else if (x < -20.0f) {
        return expf(x);
    }
    return log1pf(expf(x));
}

__global__ void optimized_softplus_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const float x = input[idx];
        output[idx] = compute_softplus_device(x);
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 512; // Opting for larger thread count for potentially better performance
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        optimized_softplus_kernel<<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}