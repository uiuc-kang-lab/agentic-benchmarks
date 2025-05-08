#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// Device function for contiguous reduction with potential vectorized loads

template <typename scalar_t>
__device__ __forceinline__ scalar_t reduce_sum_contiguous(const scalar_t* __restrict__ data, int dim_size) {
    scalar_t sum = static_cast<scalar_t>(0);

    if constexpr (std::is_same<scalar_t, float>::value) {
        // Try using float4 for 128-bit aligned loads
        if ((dim_size % 4 == 0) && (((uintptr_t)data & 0xF) == 0)) {
            int vec_iters = dim_size / 4;
            #pragma unroll
            for (int i = 0; i < vec_iters; i++) {
                float4 val = __ldg(reinterpret_cast<const float4*>(data) + i);
                sum += val.x + val.y + val.z + val.w;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < dim_size; i++) {
                sum += __ldg(data + i);
            }
        }
    } else if constexpr (std::is_same<scalar_t, double>::value) {
        // Try using double2 for 16-byte aligned loads
        if ((dim_size % 2 == 0) && (((uintptr_t)data & 0xF) == 0)) {
            int vec_iters = dim_size / 2;
            #pragma unroll
            for (int i = 0; i < vec_iters; i++) {
                double2 val = __ldg(reinterpret_cast<const double2*>(data) + i);
                sum += val.x + val.y;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < dim_size; i++) {
                sum += __ldg(data + i);
            }
        }
    } else {
        // Fallback for other types
        #pragma unroll
        for (int i = 0; i < dim_size; i++) {
            sum += __ldg(data + i);
        }
    }
    return sum;
}

// Device function for strided (non-contiguous) reduction

template <typename scalar_t>
__device__ __forceinline__ scalar_t reduce_sum_strided(const scalar_t* __restrict__ data, int dim_size, int stride) {
    scalar_t sum = static_cast<scalar_t>(0);
    for (int i = 0; i < dim_size; i++) {
        sum += __ldg(data + i * stride);
    }
    return sum;
}

// Main kernel that dispatches to the correct reduction

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * inner_size;
    if (tid >= total) return;

    int outer_idx = tid / inner_size;
    int inner_idx = tid % inner_size;
    // Offset for the start of the reduction
    int64_t offset = outer_idx * (dim_size * inner_size) + inner_idx;

    scalar_t sum = static_cast<scalar_t>(0);

    if (inner_size == 1) {
        // The reduction dimension is contiguous
        sum = reduce_sum_contiguous(input + offset, dim_size);
    } else {
        // Non-contiguous access with given stride
        sum = reduce_sum_strided(input + offset, dim_size, inner_size);
    }

    output[tid] = sum / static_cast<scalar_t>(dim_size);
}

// Host function that sets up tensor shapes and launches the kernel

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Adjust negative dimension
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    // Compute outer_size and inner_size
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Remove the reduction dimension from output shape
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Efficient mean reduction (CUDA)");
}
