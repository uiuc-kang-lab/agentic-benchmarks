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
    
    // 2D thread indexing for better thread utilization
    int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int inner_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    int64_t out_index = outer_idx * inner_size + inner_idx;
    int64_t offset = outer_idx * dim_size * inner_size + inner_idx;
    
    scalar_t sum = 0;
    
    // Vectorized loads for contiguous dimension reduction
    if (inner_size == 1) {
        if constexpr (std::is_same<scalar_t, float>::value) {
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
            for (int i = 0; i < dim_size; i++) {
                sum += __ldg(input + offset + i);
            }
        }
    } else {
        // Non-contiguous dimension reduction with strided access
        for (int i = 0; i < dim_size; i++) {
            sum += __ldg(input + offset + i * inner_size);
        }
    }
    
    output[out_index] = sum / static_cast<scalar_t>(dim_size);
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
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
    
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());
    
    // 2D block and grid configuration
    dim3 block(16, 16);
    dim3 grid((outer_size + block.x - 1) / block.x, 
              (inner_size + block.y - 1) / block.y);
              
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<grid, block>>>(
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
    m.def("forward", &mean_reduce_cuda, "Optimized mean reduction (CUDA)");
}