#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// Unified device function that performs reduction with vectorized loads when possible
template <typename scalar_t>
__device__ inline scalar_t reduce_sum(const scalar_t* data, int dim_size, int stride) {
    scalar_t sum = static_cast<scalar_t>(0);

    // When memory is contiguous along the reduction axis
    if (stride == 1) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            if ((dim_size % 4 == 0) && (((uintptr_t)data & 0xF) == 0)) {
                int vec_iters = dim_size / 4;
                for (int i = 0; i < vec_iters; i++) {
                    float4 val = __ldg(reinterpret_cast<const float4*>(data) + i);
                    sum += val.x + val.y + val.z + val.w;
                }
            } else {
                for (int i = 0; i < dim_size; i++) {
                    sum += __ldg(data + i);
                }
            }
        } else if constexpr (std::is_same<scalar_t, double>::value) {
            if ((dim_size % 2 == 0) && (((uintptr_t)data & 0xF) == 0)) {
                int vec_iters = dim_size / 2;
                for (int i = 0; i < vec_iters; i++) {
                    double2 val = __ldg(reinterpret_cast<const double2*>(data) + i);
                    sum += val.x + val.y;
                }
            } else {
                for (int i = 0; i < dim_size; i++) {
                    sum += __ldg(data + i);
                }
            }
        } else {
            for (int i = 0; i < dim_size; i++) {
                sum += __ldg(data + i);
            }
        }
    } else {
        // For non-contiguous reductions, use strided accesses
        for (int i = 0; i < dim_size; i++) {
            sum += __ldg(data + i * stride);
        }
    }
    return sum;
}

// Combined kernel that uses the unified reduce_sum function
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
    // Calculate the offset into the flattened input
    int64_t offset = outer_idx * (dim_size * inner_size) + inner_idx;

    // Use the unified reduction function
    scalar_t sum = reduce_sum(input + offset, dim_size, inner_size);
    output[tid] = sum / static_cast<scalar_t>(dim_size);
}

// Host function to setup tensor dimensions and launch the kernel
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    // Compute outer_size = product of dimensions before 'dim'
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    // Compute inner_size = product of dimensions after 'dim'
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Remove the reduced dimension to form the output tensor shape
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads>>> (
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
    m.def("forward", &mean_reduce_cuda, "Combined Mean reduction (CUDA)");
}
