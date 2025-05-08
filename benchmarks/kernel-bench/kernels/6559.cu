#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Manual unrolling of the reduction loop in groups of 4
template <typename scalar_t>
__global__ void max_reduce_manual_unroll_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = outer_size * inner_size;
    if (idx >= total_elements) return;

    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    // Compute starting index for this reduction
    int64_t base = outer_idx * dim_size * inner_size + inner_idx;

    // Initialize max_val with the first element
    scalar_t max_val = input[base];

    // Number of reduction elements to process (starting from index 1)
    int count = dim_size - 1;
    int i = 1;

    // Manually unroll reduction loop in steps of 4
    int unroll_count = count / 4;
    int remainder = count % 4;
    
    for (int j = 0; j < unroll_count; j++) {
        scalar_t a = input[base + (i + 0) * inner_size];
        scalar_t b = input[base + (i + 1) * inner_size];
        scalar_t c = input[base + (i + 2) * inner_size];
        scalar_t d = input[base + (i + 3) * inner_size];
        max_val = max(max_val, a);
        max_val = max(max_val, b);
        max_val = max(max_val, c);
        max_val = max(max_val, d);
        i += 4;
    }

    // Process any remaining iterations
    for (int j = 0; j < remainder; j++) {
        max_val = max(max_val, input[base + (i + j) * inner_size]);
    }

    output[idx] = max_val;
}

// CUDA forward function
torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);

    // Prepare output dimensions by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        max_reduce_manual_unroll_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA)");
}
