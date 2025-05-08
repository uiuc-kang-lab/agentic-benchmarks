#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename scalar_t>
__global__ void hardtanh_kernel_optimized(const scalar_t* __restrict__ x,
                                         scalar_t* __restrict__ out,
                                         int64_t numel,
                                         scalar_t min_val,
                                         scalar_t max_val) {
    // Use vectorized loads/stores where possible
    using Vec4 = typename cuda::aligned_vector<scalar_t, 4>::type;
    
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    const int64_t vec_size = sizeof(Vec4) / sizeof(scalar_t);
    const int64_t vec_numel = numel / vec_size;

    // Vector processing
    if (idx < vec_numel) {
        for (int64_t i = idx; i < vec_numel; i += stride) {
            Vec4 vec_val = *reinterpret_cast<const Vec4*>(&x[i * vec_size]);
            
            #pragma unroll
            for (int j = 0; j < vec_size; j++) {
                scalar_t val = reinterpret_cast<scalar_t*>(&vec_val)[j];
                reinterpret_cast<scalar_t*>(&vec_val)[j] = 
                    val < min_val ? min_val : (val > max_val ? max_val : val);
            }
            
            *reinterpret_cast<Vec4*>(&out[i * vec_size]) = vec_val;
        }
    }

    // Handle remaining elements
    for (int64_t i = idx + vec_numel * vec_size; i < numel; i += stride) {
        scalar_t val = x[i];
        out[i] = val < min_val ? min_val : (val > max_val ? max_val : val);
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();

    // Optimize block size based on occupancy
    const int threads = 256;
    const int blocks = std::min(65535, (numel + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        hardtanh_kernel_optimized<scalar_t><<<blocks, threads>>>(
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