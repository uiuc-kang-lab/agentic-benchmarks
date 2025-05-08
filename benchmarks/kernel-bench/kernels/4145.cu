#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

// Declare constant memory for threshold values
__constant__ float d_min_val;
__constant__ float d_max_val;
__constant__ double d_min_val_double;
__constant__ double d_max_val_double;

template <typename scalar_t>
__global__ void hardtanh_kernel_const(const scalar_t* __restrict__ x,
                                    scalar_t* __restrict__ out,
                                    int64_t numel) {
    constexpr int VecWidth = (sizeof(scalar_t) == 4 ? 4 : (sizeof(scalar_t) == 8 ? 2 : 1));
    
    // Get threshold values from constant memory based on type
    scalar_t min_val = sizeof(scalar_t) == 4 ? d_min_val : d_min_val_double;
    scalar_t max_val = sizeof(scalar_t) == 4 ? d_max_val : d_max_val_double;
    
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Vectorized processing
    if constexpr (VecWidth > 1) {
        using vec_t = typename std::conditional<sizeof(scalar_t) == 4, float4,
                        typename std::conditional<sizeof(scalar_t) == 8, double2, scalar_t>::type>::type;
                        
        const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);
        vec_t* out_vec = reinterpret_cast<vec_t*>(out);
        int64_t vecNum = numel / VecWidth;

        for (int64_t i = tid; i < vecNum; i += stride) {
            vec_t v = __ldg(&x_vec[i]);
            
            if constexpr (sizeof(scalar_t) == 4) {
                v.x = v.x < min_val ? min_val : (v.x > max_val ? max_val : v.x);
                v.y = v.y < min_val ? min_val : (v.y > max_val ? max_val : v.y);
                v.z = v.z < min_val ? min_val : (v.z > max_val ? max_val : v.z);
                v.w = v.w < min_val ? min_val : (v.w > max_val ? max_val : v.w);
            } else {
                v.x = v.x < min_val ? min_val : (v.x > max_val ? max_val : v.x);
                v.y = v.y < min_val ? min_val : (v.y > max_val ? max_val : v.y);
            }
            
            out_vec[i] = v;
        }
    }

    // Handle remaining elements
    int64_t start = (numel / VecWidth) * VecWidth;
    for (int64_t i = start + tid; i < numel; i += stride) {
        scalar_t val = __ldg(&x[i]);
        out[i] = val < min_val ? min_val : (val > max_val ? max_val : val);
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();

    // Copy threshold values to constant memory
    cudaMemcpyToSymbol(d_min_val, &min_val, sizeof(float));
    cudaMemcpyToSymbol(d_max_val, &max_val, sizeof(float));
    double min_val_double = static_cast<double>(min_val);
    double max_val_double = static_cast<double>(max_val);
    cudaMemcpyToSymbol(d_min_val_double, &min_val_double, sizeof(double));
    cudaMemcpyToSymbol(d_max_val_double, &max_val_double, sizeof(double));

    const int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, static_cast<int64_t>(1024));
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_const", ([&] {
        hardtanh_kernel_const<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &forward, "HardTanh with constant memory (CUDA)");
}