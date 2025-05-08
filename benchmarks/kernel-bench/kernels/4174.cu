#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// Optimized kernel that combines the beneficial aspects of both implementations
template <typename scalar_t>
__global__ void optimized_hardtanh_kernel(const scalar_t* __restrict__ x,
                                          scalar_t* __restrict__ out,
                                          int64_t numel,
                                          scalar_t min_val,
                                          scalar_t max_val) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t val = x[idx];
        // Direct clamping
        val = fmaxf(fminf(val, max_val), min_val);
        out[idx] = val;
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();
    
    // Use a single stream and optimal threads/block configuration
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_hardtanh_cuda", ([&] {
        optimized_hardtanh_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel,
            static_cast<scalar_t>(min_val),
            static_cast<scalar_t>(max_val)
        );
    }));

    // Synchronize to ensure kernel execution
    cudaDeviceSynchronize();

    return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
    if (!x.is_cuda()) {
        throw std::invalid_argument("Input tensor must be a CUDA tensor");
    }
    return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized HardTanh activation (CUDA)");
}