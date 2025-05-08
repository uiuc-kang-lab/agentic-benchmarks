#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel with load balancing to distribute workloads evenly across threads and blocks

template <typename scalar_t>
__global__ void load_balanced_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Calculate global thread index
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread index
    int total_threads = gridDim.x * blockDim.x;
    int total_elements = outer_size * inner_size;

    // Each thread processes multiple elements to balance the load
    for (int idx = global_idx; idx < total_elements; idx += total_threads) {
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;

        const int64_t input_offset = outer_idx * dim_size * inner_size + inner_idx;
        scalar_t max_val = input[input_offset];

        // Loop over the reduction dimension
        for (int i = 1; i < dim_size; i++) {
            scalar_t val = input[input_offset + i * inner_size];
            max_val = max(max_val, val);
        }

        // Store result in output
        output[outer_idx * inner_size + inner_idx] = max_val;
    }
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
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

    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        load_balanced_max_reduce_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Load Balanced Max Reduction Forward (CUDA)");
}
