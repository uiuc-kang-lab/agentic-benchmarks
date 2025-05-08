#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

template <typename scalar_t>
__global__ void hardtanh_kernel_hybrid(const scalar_t* __restrict__ x,
                                      scalar_t* __restrict__ out,
                                      int64_t numel,
                                      scalar_t min_val,
                                      scalar_t max_val,
                                      unsigned int* clamp_stats) {
    constexpr int VecWidth = (sizeof(scalar_t) == 4 ? 4 : (sizeof(scalar_t) == 8 ? 2 : 1));
    
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int lane_id = threadIdx.x % warpSize;
    
    int clamped_count = 0;
    int64_t vecNum = numel / VecWidth;
    
    if constexpr (VecWidth > 1) {
        using vec_t = typename std::conditional<sizeof(scalar_t) == 4, float4,
                        typename std::conditional<sizeof(scalar_t) == 8, double2, scalar_t>::type>::type;
        const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);
        vec_t* out_vec = reinterpret_cast<vec_t*>(out);

        for (int64_t i = tid; i < vecNum; i += stride) {
            vec_t v = __ldg(&x_vec[i]);
            
            if constexpr (sizeof(scalar_t) == 4) {
                if (v.x < min_val) { v.x = min_val; clamped_count++; }
                else if (v.x > max_val) { v.x = max_val; clamped_count++; }
                
                if (v.y < min_val) { v.y = min_val; clamped_count++; }
                else if (v.y > max_val) { v.y = max_val; clamped_count++; }
                
                if (v.z < min_val) { v.z = min_val; clamped_count++; }
                else if (v.z > max_val) { v.z = max_val; clamped_count++; }
                
                if (v.w < min_val) { v.w = min_val; clamped_count++; }
                else if (v.w > max_val) { v.w = max_val; clamped_count++; }
            } else {
                if (v.x < min_val) { v.x = min_val; clamped_count++; }
                else if (v.x > max_val) { v.x = max_val; clamped_count++; }
                
                if (v.y < min_val) { v.y = min_val; clamped_count++; }
                else if (v.y > max_val) { v.y = max_val; clamped_count++; }
            }
            out_vec[i] = v;
        }
    }
    
    int64_t start = vecNum * VecWidth;
    for (int64_t i = start + tid; i < numel; i += stride) {
        scalar_t val = __ldg(&x[i]);
        if (val < min_val) {
            val = min_val;
            clamped_count++;
        } else if (val > max_val) {
            val = max_val;
            clamped_count++;
        }
        out[i] = val;
    }
    
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        clamped_count += __shfl_down_sync(mask, clamped_count, offset);
    }
    
    if (lane_id == 0) {
        atomicAdd(clamp_stats, clamped_count);
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val, at::Tensor& clamp_stats) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();
    
    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda_hybrid", ([&] {
        hardtanh_kernel_hybrid<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel,
            static_cast<scalar_t>(min_val),
            static_cast<scalar_t>(max_val),
            clamp_stats.data_ptr<unsigned int>()
        );
    }));
    
    return out;
}

std::tuple<at::Tensor, at::Tensor> forward(const at::Tensor& x, float min_val, float max_val) {
    if (!x.is_cuda()) {
        throw std::invalid_argument("Input tensor must be a CUDA tensor");
    }
    auto clamp_stats = at::zeros({1}, x.options().dtype(at::kInt));
    auto output = forward_cuda(x, min_val, max_val, clamp_stats);
    return std::make_tuple(output, clamp_stats);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardTanh activation with hybrid optimization (CUDA)");
}