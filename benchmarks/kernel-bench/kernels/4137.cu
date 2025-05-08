#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

// This kernel evenly distributes the workload among threads by precomputing
// the exact range of indices each thread should process. It partitions the work
// into a vectorized portion (for types that can benefit from 128-bit loads/stores)
// and a scalar portion for any remaining elements. This ensures a balanced load
// across threads and blocks, reducing bottlenecks and underutilization.

template <typename scalar_t>
__global__ void hardtanh_kernel_evenload(const scalar_t* __restrict__ x,
                                           scalar_t* __restrict__ out,
                                           int64_t numel,
                                           scalar_t min_val,
                                           scalar_t max_val) {
    int totalThreads = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine vector width: number of scalar elements in 128 bits
    constexpr int VecWidth = (sizeof(scalar_t) == 4 ? 4 : (sizeof(scalar_t) == 8 ? 2 : 1));

    if constexpr (VecWidth > 1) {
        // Partition the vectorized portion
        int64_t vecTotal = numel / VecWidth;
        int64_t perThreadVec = vecTotal / totalThreads;
        int64_t remVec = vecTotal % totalThreads;
        int64_t startVec = tid * perThreadVec + (tid < remVec ? tid : remVec);
        int64_t countVec = perThreadVec + (tid < remVec ? 1 : 0);

        // Define vector type for 128-bit loads/stores
        using vec_t = typename std::conditional<sizeof(scalar_t) == 4, float4,
                        typename std::conditional<sizeof(scalar_t) == 8, double2, scalar_t>::type>::type;
        const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);
        vec_t* out_vec = reinterpret_cast<vec_t*>(out);

        for (int64_t i = startVec; i < startVec + countVec; i++) {
            vec_t v = __ldg(&x_vec[i]);
            if constexpr (sizeof(scalar_t) == 4) {
                v.x = (v.x < min_val ? min_val : (v.x > max_val ? max_val : v.x));
                v.y = (v.y < min_val ? min_val : (v.y > max_val ? max_val : v.y));
                v.z = (v.z < min_val ? min_val : (v.z > max_val ? max_val : v.z));
                v.w = (v.w < min_val ? min_val : (v.w > max_val ? max_val : v.w));
            } else if constexpr (sizeof(scalar_t) == 8) {
                v.x = (v.x < min_val ? min_val : (v.x > max_val ? max_val : v.x));
                v.y = (v.y < min_val ? min_val : (v.y > max_val ? max_val : v.y));
            }
            out_vec[i] = v;
        }

        // Process the remaining scalar elements
        int64_t startScalar = vecTotal * VecWidth;
        int64_t scalarTotal = numel - startScalar;
        int64_t perThreadScalar = scalarTotal / totalThreads;
        int64_t remScalar = scalarTotal % totalThreads;
        int64_t startS = startScalar + tid * perThreadScalar + (tid < remScalar ? tid : remScalar);
        int64_t countS = perThreadScalar + (tid < remScalar ? 1 : 0);
        for (int64_t i = startS; i < startS + countS; i++) {
            scalar_t val = __ldg(&x[i]);
            out[i] = (val < min_val ? min_val : (val > max_val ? max_val : val));
        }
    } else {
        // Fallback for scalar types (VecWidth == 1): evenly distribute all elements
        int64_t perThread = numel / totalThreads;
        int64_t rem = numel % totalThreads;
        int64_t start = tid * perThread + (tid < rem ? tid : rem);
        int64_t count = perThread + (tid < rem ? 1 : 0);
        for (int64_t i = start; i < start + count; i++) {
            scalar_t val = __ldg(&x[i]);
            out[i] = (val < min_val ? min_val : (val > max_val ? max_val : val));
        }
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();

    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda_evenload", ([&] {
        hardtanh_kernel_evenload<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &forward, "HardTanh activation with even load distribution (CUDA)");
}
