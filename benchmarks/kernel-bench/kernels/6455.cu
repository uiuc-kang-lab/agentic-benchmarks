#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// CUDA kernel using grid-stride loops for handling workloads larger than the number of threads
template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    int64_t total_outputs = outer_size * inner_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Use a grid-stride loop to process multiple output elements per thread
    for (int tid = idx; tid < total_outputs; tid += stride) {
        int outer_idx = tid / inner_size;
        int inner_idx = tid % inner_size;
        // Compute the starting offset for the reduction dimension
        int64_t offset = outer_idx * (dim_size * inner_size) + inner_idx;

        scalar_t sum = static_cast<scalar_t>(0);
        if (inner_size == 1) {
            // Contiguous reduction along the reduction dimension: attempt vectorized loads
            if constexpr (std::is_same<scalar_t, float>::value) {
                if ((dim_size % 4 == 0) && ((((uintptr_t)(input + offset)) & 0xF) == 0)) {
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
                if ((dim_size % 2 == 0) && ((((uintptr_t)(input + offset)) & 0xF) == 0)) {
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
            // Non-contiguous reduction: load with the proper stride
            for (int i = 0; i < dim_size; i++) {
                sum += __ldg(input + offset + i * inner_size);
            }
        }
        
        output[tid] = sum / static_cast<scalar_t>(dim_size);
    }
}

// Host function launching the kernel
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();
    
    // Retrieve the tensor dimensions
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];
    
    // Compute outer and inner sizes
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }
    
    // Create output tensor by removing the reduction dimension
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());
    
    // Launch the kernel using grid-stride loop for larger workloads
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA) with grid-stride loops");
}
