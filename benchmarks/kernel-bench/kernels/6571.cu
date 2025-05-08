#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Ensure memory coalescing by aligning global memory accesses.
template <typename scalar_t>
__global__ void coalesced_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t outer_size
) {
    // Each block processes multiple output elements in a grid-stride loop.
    for (int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (int inner_idx = blockIdx.y * blockDim.x + threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x * gridDim.y) {

            int64_t start_idx = outer_idx * dim_size * inner_size + inner_idx;
            scalar_t max_val = input[start_idx];

            // Perform coalesced memory access for reduction along dim
            for (int i = 1; i < dim_size; ++i) {
                max_val = max(max_val, input[start_idx + i * inner_size]);
            }

            output[outer_idx * inner_size + inner_idx] = max_val;
        }
    }
}

// CUDA forward function
torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++)
        outer_size *= input.size(i);
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++)
        inner_size *= input.size(i);

    const int64_t dim_size = input.size(dim);

    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    int block_size = 256;
    dim3 threads(block_size);
    dim3 blocks((outer_size + threads.x - 1) / threads.x, (inner_size + threads.x - 1) / threads.x);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "coalesced_max_reduce_forward", ([&] {
        coalesced_max_reduce_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Coalesced max reduce forward (CUDA)");
}