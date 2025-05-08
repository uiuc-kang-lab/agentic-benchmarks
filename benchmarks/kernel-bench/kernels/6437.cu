#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>


template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    int outer_idx = tid / inner_size;
    int inner_idx = tid % inner_size;
    // Calculate the starting offset for this thread
    int64_t offset = outer_idx * dim_size * inner_size + inner_idx;

    scalar_t sum = 0;

    // If the reduction dimension is contiguous, try to apply 128-bit vectorized loads
    if (inner_size == 1) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            // float: try to load 4 floats at a time using float4
            if ((dim_size % 4 == 0) && (((uintptr_t)(input + offset) & 0xF) == 0)) {
                int vec_iters = dim_size / 4;
                for (int i = 0; i < vec_iters; i++) {
                    float4 val = __ldg(reinterpret_cast<const float4*>(input + offset) + i);
                    sum += val.x + val.y + val.z + val.w;
                }
            } else {
                for (int i = 0; i < dim_size; i++) {
                    sum += __ldg(input + offset + i);
                }
            }
        } else if constexpr (std::is_same<scalar_t, double>::value) {
            // double: load 2 doubles at a time using double2 (16 bytes)
            if ((dim_size % 2 == 0) && (((uintptr_t)(input + offset) & 0xF) == 0)) {
                int vec_iters = dim_size / 2;
                for (int i = 0; i < vec_iters; i++) {
                    double2 val = __ldg(reinterpret_cast<const double2*>(input + offset) + i);
                    sum += val.x + val.y;
                }
            } else {
                for (int i = 0; i < dim_size; i++) {
                    sum += __ldg(input + offset + i);
                }
            }
        } else {
            // Fallback for other types: scalar loads with __ldg()
            for (int i = 0; i < dim_size; i++) {
                sum += __ldg(input + offset + i);
            }
        }
    } else {
        // When the reduction dimension is not contiguous, use scalar __ldg() loads with proper striding
        for (int i = 0; i < dim_size; i++) {
            sum += __ldg(input + offset + i * inner_size);
        }
    }

    output[tid] = sum / static_cast<scalar_t>(dim_size);
}


torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimension
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

    // Erase the reduction dimension
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    // Launch kernel
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA)");
}
