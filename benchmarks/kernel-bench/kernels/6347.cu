#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory for intra-block reduction and warp-level primitives
// for the final stage of reduction. Each block computes one output element by summing
// over the reduction dimension. Threads in the block accumulate partial sums from the input
// and store them in shared memory. The shared memory is reduced to 32 elements, then the final
// reduction is performed using __shfl_down_sync to minimize synchronization overhead.

template <typename scalar_t>
__global__ void shared_mem_warp_reduce_sum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {

    // Each block corresponds to one output element
    int out_idx = blockIdx.x;  
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    // Each thread computes a partial sum over the reduction dimension
    scalar_t sum = 0;
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        sum += input[base + i * inner_size];
    }

    // Allocate shared memory dynamically
    extern __shared__ scalar_t sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Perform reduction in shared memory: reduce blockDim.x elements down to 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Final warp-level reduction using shuffle instructions
    if (threadIdx.x < 32) {
        // Load the remaining data into a register
        scalar_t val = sdata[threadIdx.x];
        // Use warp-level primitives to finalize the reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        // The first thread writes the final output
        if (threadIdx.x == 0) {
            output[out_idx] = val;
        }
    }
}

// CUDA wrapper function
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    // Compute the outer and inner dimensions
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Prepare the output tensor by setting the reduced dimension to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Each output element will be computed by one block
    int64_t total_outputs = outer_size * inner_size;
    int blocks = total_outputs;
    int threads = 256;  // Using 256 threads per block

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        size_t shared_mem_size = threads * sizeof(scalar_t);
        shared_mem_warp_reduce_sum_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Shared memory warp-level sum reduction forward (CUDA)");
}
