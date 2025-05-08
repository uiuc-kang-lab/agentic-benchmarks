#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that distributes the workload evenly by processing multiple output elements per block
// using a 2D thread block. Threads along the x-dimension handle parts of the reduction dimension,
// while the y-dimension indexes different output elements within the block. A shared memory reduction
// is performed for each output element.

template <typename scalar_t>
__global__ void block_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_output) {

    // 2D block: threadIdx.x handles the reduction dimension; threadIdx.y indexes output elements
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute the global output index based on block and thread indices
    int out_idx = blockIdx.x * blockDim.y + ty;
    if (out_idx >= total_output) return;

    // Map the 1D output index to its corresponding (outer, inner) indices
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;

    // Each thread computes a partial sum over the reduction dimension
    scalar_t partial = 0;
    for (int i = tx; i < reduce_size; i += blockDim.x) {
        int64_t offset = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        partial += input[offset];
    }

    // Allocate shared memory dynamically for the block
    extern __shared__ scalar_t sdata[];
    int sindex = ty * blockDim.x + tx;
    sdata[sindex] = partial;
    __syncthreads();

    // Reduce the partial sums in shared memory along the x-dimension
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            sdata[sindex] += sdata[sindex + stride];
        }
        __syncthreads();
    }

    // The first thread in the x-dimension writes the final sum to the output
    if (tx == 0) {
        output[out_idx] = sdata[ty * blockDim.x];
    }
}

// Host function wrapping the CUDA kernel and setting up grid/block dimensions

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    // Compute outer_size (product of dimensions before 'dim')
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    // Compute inner_size (product of dimensions after 'dim')
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Prepare output tensor with the reduced dimension set to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements
    int64_t total_output = outer_size * inner_size;

    // Configure a 2D block: threads in x handle the reduction dimension,
    // and threads in y allow multiple output elements to be processed per block
    dim3 block(128, 8);
    int grid_x = (total_output + block.y - 1) / block.y;
    dim3 grid(grid_x);

    // Allocate shared memory: one element for each thread in the block
    size_t shared_mem_size = block.x * block.y * sizeof(at::cuda::detail::scalar_t);
    // Instead of at::cuda::detail::scalar_t, we use the type from AT_DISPATCH below
    shared_mem_size = block.x * block.y * sizeof(float); // This line is a placeholder; actual size computed in AT_DISPATCH

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        size_t smem = block.x * block.y * sizeof(scalar_t);
        block_sum_reduce_kernel<scalar_t><<<grid, block, smem>>>(
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
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA) using 2D block occupancy");
}
