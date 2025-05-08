#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel assigns each block to an outer slice to ensure that all threads in the block work on the same outer index.
// This allows threads to read consecutive (coalesced) memory locations along the inner dimension for each step of the reduction.

template <typename scalar_t>
__global__ void sum_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t reduce_size,
    int64_t inner_size) {

    // Each block processes one outer index
    int outer_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Compute the starting offset for this outer index
    int64_t base_offset = outer_idx * reduce_size * inner_size;
    int64_t out_offset = outer_idx * inner_size;

    // Each thread processes a subset of the inner dimension
    for (int inner_idx = tid; inner_idx < inner_size; inner_idx += blockDim.x) {
        scalar_t sum = 0;
        // Sum across the reduction dimension; note that for each fixed i, accesses are consecutive in memory.
        for (int i = 0; i < reduce_size; i++) {
            sum += input[base_offset + i * inner_size + inner_idx];
        }
        output[out_offset + inner_idx] = sum;
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
    int threads = inner_size < 1024 ? inner_size : 1024;
    int blocks = outer_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        sum_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA)");
}
