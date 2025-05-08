#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

template <typename scalar_t>
__global__ void hardtanh_kernel_tuned(const scalar_t* __restrict__ x,
                                     scalar_t* __restrict__ out,
                                     int64_t numel,
                                     scalar_t min_val,
                                     scalar_t max_val) {
    constexpr int VecWidth = (sizeof(scalar_t) == 4 ? 4 : (sizeof(scalar_t) == 8 ? 2 : 1));
    
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    int64_t vecNum = numel / VecWidth;

    if constexpr (VecWidth > 1) {
        using vec_t = typename std::conditional<sizeof(scalar_t) == 4, float4,
                        typename std::conditional<sizeof(scalar_t) == 8, double2, scalar_t>::type>::type;
        const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);
        vec_t* out_vec = reinterpret_cast<vec_t*>(out);

        for (int64_t i = tid; i < vecNum; i += stride) {
            vec_t v = __ldg(&x_vec[i]);
            if constexpr (sizeof(scalar_t) == 4) {
                v.x = max(min_val, min(max_val, v.x));
                v.y = max(min_val, min(max_val, v.y));
                v.z = max(min_val, min(max_val, v.z));
                v.w = max(min_val, min(max_val, v.w));
            } else {
                v.x = max(min_val, min(max_val, v.x));
                v.y = max(min_val, min(max_val, v.y));
            }
            out_vec[i] = v;
        }
    }

    int64_t start = vecNum * VecWidth;
    for (int64_t i = start + tid; i < numel; i += stride) {
        scalar_t val = __ldg(&x[i]);
        out[i] = max(min_val, min(max_val, val));
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();

    const int threads = 128;
    int max_blocks = 1024;
    int blocks = std::min((numel + threads - 1) / threads, static_cast<int64_t>(max_blocks));
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        hardtanh_kernel_tuned<scalar_t><<<blocks, threads>>>(
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