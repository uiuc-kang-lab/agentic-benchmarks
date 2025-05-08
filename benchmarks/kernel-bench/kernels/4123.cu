#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename scalar_t>
__global__ void optimized_hardtanh_kernel(const scalar_t* __restrict__ x,
                                         scalar_t* __restrict__ out,
                                         int64_t numel,
                                         scalar_t min_val,
                                         scalar_t max_val) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Vector loading for better memory bandwidth utilization
    using Vec4 = typename std::aligned_storage<sizeof(float4), alignof(float4)>::type;
    const Vec4* x4 = reinterpret_cast<const Vec4*>(x);
    Vec4* out4 = reinterpret_cast<Vec4*>(out);
    
    // Handle vector loads for aligned data
    int64_t vec_elements = (numel / 4) * 4;
    for (int64_t i = tid; i < vec_elements/4; i += stride) {
        float4 val4 = reinterpret_cast<const float4*>(x4)[i];
        
        // Process all 4 elements
        val4.x = max(min_val, min(max_val, val4.x));
        val4.y = max(min_val, min(max_val, val4.y));
        val4.z = max(min_val, min(max_val, val4.z));
        val4.w = max(min_val, min(max_val, val4.w));
        
        reinterpret_cast<float4*>(out4)[i] = val4;
    }
    
    // Handle remaining elements
    for (int64_t i = tid + vec_elements; i < numel; i += stride) {
        scalar_t val = x[i];
        out[i] = max(min_val, min(max_val, val));
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();

    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        optimized_hardtanh_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &forward, "HardTanh activation (CUDA)");
}