#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel eliminates shared memory usage by using only warp-level primitives
// Each CUDA block is set to exactly one warp (32 threads) so that the entire reduction
// for one output element can be performed using __shfl_down_sync without shared memory.

template <typename scalar_t>
__global__ void sum_reduce_warp_no_shmem_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t num_output) {

    // Each block corresponds to one output element in the flattened (outer * inner) dimensions
    int out_idx = blockIdx.x;  // flattened index over outer_size * inner_size
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;

    // Compute the base index for the reduction segment
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    scalar_t sum = 0;
    // Loop over the reduction dimension with a stride equal to warp size (32)
    // Each thread sums a subset of the elements.
    for (int i = threadIdx.x; i < reduce_size; i += 32) {
        sum += input[base + i * inner_size];
    }

    // Perform warp-level reduction using shuffle intrinsics.
    // All threads in the warp participate in the reduction with full mask.
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // The first thread in the warp writes the result to global memory
    if (threadIdx.x == 0) {
        output[out_idx] = sum;
    }
}

// Host function to prepare and launch the kernel
// For each output element (flattened across outer and inner dimensions), we launch one block of 32 threads

torch::Tensor sum_reduce_warp_no_shmem_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dim
    if (dim < 0) dim += input.dim();

    // Compute outer, reduce and inner sizes
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

    // Set output tensor shape: collapse the reduction dimension to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements
    int64_t num_output = outer_size * inner_size;

    // Launch one block per output element with 32 threads per block
    int threads = 32;
    int blocks = num_output;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_warp_no_shmem_cuda", ([&] {
        sum_reduce_warp_no_shmem_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_warp_no_shmem_cuda, "Sum reduction without shared memory using warp-level primitives (CUDA)");
}
