#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: each block computes the sum reduction for one output element
// using shared memory to aggregate partial sums from its threads.

template <typename scalar_t>
__global__ void sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {

    extern __shared__ scalar_t sdata[];  // shared memory for partial sums
    int tid = threadIdx.x;

    // Each block is responsible for one output element
    int out_idx = blockIdx.x;  // unique index for each output element
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    
    // Compute base offset in input tensor for this output element
    int64_t base_offset = outer_idx * reduce_size * inner_size + inner_idx;

    // Each thread accumulates a partial sum over a strided portion of the reduce dimension
    scalar_t sum = 0;
    for (int i = tid; i < reduce_size; i += blockDim.x) {
        sum += input[base_offset + i * inner_size];
    }

    // Store each thread's partial sum into shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Perform tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread writes the block's result to the output
    if (tid == 0) {
        output[out_idx] = sdata[0];
    }
}


// CUDA wrapper function
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Compute sizes for reduction
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

    // Prepare the output tensor (reduced dimension is set to 1)
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Each output element is computed by one block
    int blocks = outer_size * inner_size;
    // Choose a block size tuned for the GPU (e.g., 256 threads per block)
    int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        sum_reduce_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction over a dimension using block-level shared memory reduction (CUDA)");
}
