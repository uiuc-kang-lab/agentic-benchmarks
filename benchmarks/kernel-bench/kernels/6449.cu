/*
 * Combined CUDA Kernel for Mean Reduction
 * This kernel uses a 2D grid mapping to assign each (outer, inner) coordinate to a thread.
 * When the reduction dimension is contiguous (i.e., inner_size == 1), it exploits vectorized loads
 * (using float4 for floats and double2 for doubles) to improve memory bandwidth utilization.
 * Otherwise, it falls back to scalar loads with proper striding.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>


// Combined kernel template
template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t dim_size,
    int64_t inner_size,
    int64_t outer_size) {

    // 2D grid: blockIdx.x/threadIdx.x for outer dimension, blockIdx.y/threadIdx.y for inner dimension
    int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int inner_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        // Calculate output index (reduction dimension removed)
        int64_t out_index = outer_idx * inner_size + inner_idx;
        // Compute starting offset for reduction along the specified dim
        int64_t offset = outer_idx * dim_size * inner_size + inner_idx;
        scalar_t sum = 0;

        // If the reduction dimension is contiguous, try to apply vectorized loads
        if (inner_size == 1) {
            if constexpr (std::is_same<scalar_t, float>::value) {
                // Check alignment and vectorization possibility
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
                // Fallback for other data types
                for (int i = 0; i < dim_size; i++) {
                    sum += __ldg(input + offset + i);
                }
            }
        } else {
            // Non-contiguous reduction dimension: use scalar loads with proper striding
            for (int i = 0; i < dim_size; i++) {
                sum += __ldg(input + offset + i * inner_size);
            }
        }

        output[out_index] = sum / static_cast<scalar_t>(dim_size);
    }
}


// CUDA wrapper function
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Retrieve tensor sizes and compute reduction parameters
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Remove the reduced dimension from output shape
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    // Define a 2D grid mapping: block and grid sizes
    // The 2D configuration efficiently maps work to threads irrespective of inner contiguity
    dim3 block(16, 16);
    dim3 grid((outer_size + block.x - 1) / block.x, (inner_size + block.y - 1) / block.y);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            inner_size,
            outer_size
        );
    }));

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Efficient mean reduction with vectorized loads and 2D grid (CUDA)");
}
