#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using both block-level and warp-level techniques for efficient reduction

template <typename scalar_t>
__global__ void optimized_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_output) {

    extern __shared__ scalar_t shared_memory[];  // Shared memory for block-level reduction

    // Each block handles one output element (one (outer, inner) pair).
    int idx = blockIdx.x;  // index for output element
    if (idx >= total_output) return;

    // Determine corresponding outer and inner indices
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;

    // Perform reduction in a strided manner
    scalar_t sum = 0;
    int tid = threadIdx.x;

    // Sum elements from the reduction dim
    for (int i = tid; i < reduce_size; i += blockDim.x) {
        int64_t offset = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        sum += input[offset];
    }

    // Store the sum in shared memory and synchronize
    shared_memory[tid] = sum;
    __syncthreads();

    // Reduce within a block using warp-level shuffle operations
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid < warpSize) {
        shared_memory[tid] = sum;
    }
    __syncthreads();

    // Final reduction of the warp results in shared memory
    if (tid == 0) {
        sum = 0;
        for (int i = 0; i < blockDim.x / warpSize; i++) {
            sum += shared_memory[i];
        }
        output[idx] = sum;
    }
}

// Host function wrapping the kernel launch

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Adjust negative dimensions
    if (dim < 0) dim += input.dim();
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    // Compute sizes for reduction operation
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Prepare output tensor with reduced dimension set to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements
    int64_t total_output = outer_size * inner_size;

    // Launch configuration
    const int threads = 256;  // Ideal block size
    const int blocks = total_output;
    const int shared_memory = threads * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        optimized_sum_reduce_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size,
            total_output
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Optimized sum reduction forward (CUDA)");
}
