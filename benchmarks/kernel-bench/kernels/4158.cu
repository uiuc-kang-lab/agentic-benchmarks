#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Optimized HardTanh kernel with workload distribution
// Calculate grid size dynamically to improve load balancing
template <typename scalar_t>
__global__ void hardtanh_kernel(const scalar_t* __restrict__ x,
                                scalar_t* __restrict__ out,
                                int64_t numel,
                                scalar_t min_val,
                                scalar_t max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += stride) {
        scalar_t val = x[i];
        // Clamp value to [min_val, max_val]
        val = (val < min_val) ? min_val : val;
        val = (val > max_val) ? max_val : val;
        out[i] = val;
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();

    const int threads = 1024;
    // Adjusting blocks for optimal GPU utilization and workload distribution
    const int blocks = (numel + threads * 4 - 1) / (threads * 4);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        hardtanh_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel,
            static_cast<scalar_t>(min_val),
            static_cast<scalar_t>(max_val)
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
    m.def("forward", &forward, "HardTanh activation optimized (CUDA)");
}