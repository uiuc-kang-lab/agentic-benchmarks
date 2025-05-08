#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Declare constant memory for min and max values
__constant__ float const_min_val;
__constant__ float const_max_val;

template <typename scalar_t>
__forceinline__ __global__ void hardtanh_const_kernel(
    const scalar_t* const __restrict__ x,
    scalar_t* const __restrict__ out,
    const int64_t numel) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t val = x[idx];
        // Direct comparison instead of using fmaxf/fminf to reduce register usage
        if (val < const_min_val) val = const_min_val;
        if (val > const_max_val) val = const_max_val;
        out[idx] = val;
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();
    
    // Copy min_val and max_val to constant memory
    cudaMemcpyToSymbol(const_min_val, &min_val, sizeof(float));
    cudaMemcpyToSymbol(const_max_val, &max_val, sizeof(float));
    
    const int threads = 512;
    const int blocks = (numel + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        hardtanh_const_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel
        );
    }));
    
    return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
    if (!x.is_cuda()) {
        throw std::invalid_argument("Input tensor must be a CUDA tensor");
    }
    return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardTanh activation with constant memory (CUDA)");
}