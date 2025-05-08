#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that performs reduction over the specified dimension using shared memory
// Each block computes one output element by collaboratively summing over the reduction dimension

template <typename scalar_t>
__global__ void sum_reduce_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {

    // Each block is responsible for one output element corresponding to a unique (outer, inner) index
    int out_idx = blockIdx.x;  // global output index
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;

    // Compute the base offset of the block's reduction segment
    // The input tensor is assumed to be laid out as: [outer_size, reduce_size, inner_size]
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    // Each thread loads a subset of the reduction elements and accumulates a partial sum
    scalar_t partial_sum = 0;
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        partial_sum += input[base + i * inner_size];
    }

    // Use dynamically allocated shared memory to reduce the partial sums from all threads
    extern __shared__ __shared__ scalar_t sdata[512];
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread writes the final sum to global memory
    if (threadIdx.x == 0) {
        output[out_idx] = sdata[0];
    }
}

// Host function that wraps the shared memory reduction kernel
// It reshapes the tensor dimensions and launches one block per output element

torch::Tensor sum_reduce_shared_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimension
    if (dim < 0) dim += input.dim();

    // Compute sizes for outer, reduce, and inner parts
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

    // The reduced dimension will collapse to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    int64_t num_output_elements = outer_size * inner_size;
    
    // Configure the grid: one block per output element
    int threads = 512;  // Experimenting with 512 threads per block
    dim3 blocks(num_output_elements);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_shared_cuda", ([&] {
        size_t smem_size = threads * sizeof(scalar_t);
        sum_reduce_shared_kernel<scalar_t><<<blocks, threads, smem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_shared_cuda, "Sum reduction with shared memory (CUDA)");
}
