#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel partitions the reduction dimension among multiple blocks for each output element.
// Each block computes a partial sum over its assigned chunk using shared memory, then uses a single atomicAdd
// to accumulate the partial result to the global output. This minimizes the usage of atomic operations to
// only one per block, reducing global memory contention.

template <typename scalar_t>
__global__ void atomic_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t n_out) {

    // blockIdx.x indexes the output element (combination of outer and inner dimensions)
    int out_idx = blockIdx.x; 
    if (out_idx >= n_out) return;

    // Decode outer and inner indices from out_idx
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;

    // Determine the chunk of the reduction dimension this block will cover.
    // gridDim.y is the number of segments each output element is divided into.
    int threadsInBlock = blockDim.x;
    int seg = blockIdx.y; 
    int chunks = gridDim.y;
    int chunk_size = (reduce_size + chunks - 1) / chunks;
    int start = seg * chunk_size;
    int end = start + chunk_size;
    if (end > reduce_size) end = reduce_size;

    scalar_t sum = 0;
    // The input tensor is assumed to be laid out as: input[outer_idx * reduce_size * inner_size + i * inner_size + inner_idx]
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    // Each thread processes a portion of the assigned chunk in a strided manner
    for (int i = start + threadIdx.x; i < end; i += threadsInBlock) {
        int64_t offset = base + i * inner_size;
        sum += input[offset];
    }

    // Perform a block-level reduction using shared memory
    extern __shared__ char smem[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(smem);
    shared[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = threadsInBlock / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the block's partial sum to the output using atomic addition
    if (threadIdx.x == 0) {
        atomicAdd(&output[out_idx], shared[0]);
    }
}

// Host function: sets up grid dimensions and launches the kernel

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Get tensor dimensions
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    // Compute outer and inner sizes
    int64_t outer = 1;
    for (int i = 0; i < dim; i++) {
        outer *= sizes[i];
    }
    int64_t inner = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
        inner *= sizes[i];
    }
    sizes[dim] = 1;  // Reduction dimension becomes 1 in the output

    // Allocate output tensor and initialize to zero for atomic accumulation
    auto output = torch::zeros(sizes, input.options());
    
    // Total number of output elements
    int64_t n_out = outer * inner;

    // Configure kernel launch parameters
    const int threads = 256;
    // Partition the reduction dimension: one segment per ~'threads' elements
    int blocks_y = (reduce_size + threads - 1) / threads;
    // Grid is 2D: x-dimension for each output element, y-dimension for segments of reduction
    dim3 grid(n_out, blocks_y);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        atomic_sum_reduce_kernel<scalar_t><<<grid, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner,
            n_out
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA) using atomic operations minimally");
}
