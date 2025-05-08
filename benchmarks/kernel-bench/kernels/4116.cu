#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename scalar_t>
__global__ void hardtanh_kernel_aligned(const scalar_t* __restrict__ x,
                                      scalar_t* __restrict__ out,
                                      int64_t numel,
                                      scalar_t min_val,
                                      scalar_t max_val) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time when possible
    const int64_t vec4_elements = numel & ~3;
    
    for (int64_t i = idx * 4; i < vec4_elements; i += stride * 4) {
        float4 in4;
        float4 out4;
        
        // Load 4 elements using __ldg
        reinterpret_cast<float4*>(&in4)->x = __ldg(&x[i]);
        reinterpret_cast<float4*>(&in4)->y = __ldg(&x[i + 1]);
        reinterpret_cast<float4*>(&in4)->z = __ldg(&x[i + 2]);
        reinterpret_cast<float4*>(&in4)->w = __ldg(&x[i + 3]);
        
        // Process each element
        out4.x = in4.x < min_val ? min_val : (in4.x > max_val ? max_val : in4.x);
        out4.y = in4.y < min_val ? min_val : (in4.y > max_val ? max_val : in4.y);
        out4.z = in4.z < min_val ? min_val : (in4.z > max_val ? max_val : in4.z);
        out4.w = in4.w < min_val ? min_val : (in4.w > max_val ? max_val : in4.w);
        
        // Store 4 elements
        reinterpret_cast<float4*>(&out[i])[0] = out4;
    }
    
    // Handle remaining elements
    for (int64_t i = idx + vec4_elements; i < numel; i += stride) {
        scalar_t val = __ldg(&x[i]);
        out[i] = val < min_val ? min_val : (val > max_val ? max_val : val);
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();

    const int threads = 256;
    const int blocks = std::min(65535, (int)((numel + threads * 4 - 1) / (threads * 4)));

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        hardtanh_kernel_aligned<scalar_t><<<blocks, threads>>>(
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