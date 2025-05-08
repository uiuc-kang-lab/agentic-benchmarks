#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Fallback scalar kernel for non-float types
template <typename scalar_t>
__global__ void hardtanh_scalar_ldg(const scalar_t* __restrict__ x,
                                    scalar_t* __restrict__ out,
                                    int64_t numel,
                                    scalar_t min_val,
                                    scalar_t max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < numel; i += stride) {
        scalar_t v = __ldg(&x[i]);
        out[i] = v < min_val ? min_val : (v > max_val ? max_val : v);
    }
}

// Optimized kernel that distributes workload evenly with manual loop unrolling by a factor of 4
__global__ void hardtanh_even_distribution_kernel(const float* __restrict__ x,
                                                    float* __restrict__ out,
                                                    int64_t numel,
                                                    float min_val,
                                                    float max_val) {
    int total_threads = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Unroll loop: process 4 elements per iteration when enough elements remain
    while (i + 3 * total_threads < numel) {
        float a = __ldg(&x[i]);
        float b = __ldg(&x[i + total_threads]);
        float c = __ldg(&x[i + 2 * total_threads]);
        float d = __ldg(&x[i + 3 * total_threads]);
        
        out[i]                     = a < min_val ? min_val : (a > max_val ? max_val : a);
        out[i + total_threads]     = b < min_val ? min_val : (b > max_val ? max_val : b);
        out[i + 2 * total_threads] = c < min_val ? min_val : (c > max_val ? max_val : c);
        out[i + 3 * total_threads] = d < min_val ? min_val : (d > max_val ? max_val : d);
        
        i += total_threads * 4;
    }
    
    // Process any remaining elements
    while (i < numel) {
        float v = __ldg(&x[i]);
        out[i] = v < min_val ? min_val : (v > max_val ? max_val : v);
        i += total_threads;
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();

    // Use 256 threads per block for good occupancy
    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_even_distribution_cuda", ([&] {
        if (std::is_same<scalar_t, float>::value) {
            hardtanh_even_distribution_kernel<<<blocks, threads>>>(
                x.data_ptr<float>(),
                out.data_ptr<float>(),
                numel,
                min_val,
                max_val
            );
        } else {
            hardtanh_scalar_ldg<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                numel,
                static_cast<scalar_t>(min_val),
                static_cast<scalar_t>(max_val)
            );
        }
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
    m.def("forward", &forward, "HardTanh CUDA optimized with even workload distribution");
}
