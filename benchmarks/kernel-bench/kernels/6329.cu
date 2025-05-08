#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel minimizes warp divergence by ensuring that all threads in a warp follow the same execution path.
// It uses a single loop to handle both the outer and inner dimensions, avoiding conditional logic within the loop.

template <typename scalar_t>
__global__ void sum_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    // Each thread processes multiple elements in a grid-stride loop
    for (int idx = tid; idx < outer_size * inner_size; idx += total_threads) {
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;

        scalar_t sum = 0;
        int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;

        // Perform reduction along the specified dimension
        for (int i = 0; i < reduce_size; i++) {
            sum += input[base_idx + i * inner_size];
        }

        output[outer_idx * inner_size + inner_idx] = sum;
    }
}

// CUDA wrapper
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Set output size: the reduction dimension becomes 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Launch configuration: one block per outer index;
    // threads per block chosen to cover the inner dimension with a maximum of 1024 threads
    int threads = 1024;
    int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        sum_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            outer_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA)");
}
