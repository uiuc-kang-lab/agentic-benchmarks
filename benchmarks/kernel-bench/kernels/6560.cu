#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses grid-stride loops to ensure uniform control flow and minimize warp divergence
template <typename scalar_t>
__global__ void max_reduce_kernel_uniform(
    const scalar_t* input,
    scalar_t* output,
    const int64_t total_elements,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Use grid-stride loop to reduce warp divergence
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        const int outer_idx = idx / inner_size;
        const int inner_idx = idx % inner_size;
        const int64_t start_idx = outer_idx * dim_size * inner_size + inner_idx;

        // Initialize with the first element
        scalar_t max_val = input[start_idx];

        // Unroll loop to reduce overhead and maintain uniform control flow
        #pragma unroll
        for (int i = 1; i < dim_size; i++) {
            scalar_t val = input[start_idx + i * inner_size];
            max_val = max(max_val, val);
        }

        output[idx] = max_val;
    }
}

// CUDA forward function
torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) {
        dim += input.dim();
    }

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    int64_t total_elements = outer_size * inner_size;
    const int64_t dim_size = input.size(dim);

    // Create output tensor with the reduced dimension removed
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        max_reduce_kernel_uniform<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_elements,
            dim_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA) with uniform warp scheduling");
}
