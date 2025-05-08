#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel leverages shared memory to load a tile of the reduction dimension from global memory,
// reducing global memory latency. Each block is responsible for one output element (corresponding
// to a unique combination of outer and inner indices). The reduction dimension is processed in
// tiles of size equal to the block width. Each tile is loaded into shared memory, reduced
// in parallel, and its result is accumulated into a register. Finally, the total sum is written
// to global memory by thread 0 of the block.

template <typename scalar_t>
__global__ void sum_reduce_shared_tile_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {

    extern __shared__ char smem[];  // dynamically allocated shared memory
    scalar_t* tile = reinterpret_cast<scalar_t*>(smem);

    // Each block computes one output element. Compute corresponding indices:
    int out_idx = blockIdx.x;  // flattened index over [outer_size * inner_size]
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    // Starting position of the reduction slice in the input
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    scalar_t sum = 0;  // accumulator for the reduction result
    int tile_size = blockDim.x;  // number of elements per shared memory tile

    // Process the reduction dimension in chunks of 'tile_size'
    for (int tile_offset = 0; tile_offset < reduce_size; tile_offset += tile_size) {
        int index = tile_offset + threadIdx.x;
        // Each thread loads one element from global memory into shared memory if within bounds
        if (index < reduce_size) {
            tile[threadIdx.x] = input[base + index * inner_size];
        } else {
            tile[threadIdx.x] = 0;
        }
        __syncthreads();

        // Perform reduction on the tile in shared memory
        for (int stride = tile_size / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                tile[threadIdx.x] += tile[threadIdx.x + stride];
            }
            __syncthreads();
        }
        // The first thread in the block adds the sum from this tile
        if (threadIdx.x == 0) {
            sum += tile[0];
        }
        __syncthreads();
    }

    // Write the final reduction result to the output tensor
    if (threadIdx.x == 0) {
        output[out_idx] = sum;
    }
}

// Host function to launch the shared tile-based sum reduction kernel

torch::Tensor sum_reduce_shared_tile_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimension indices
    if (dim < 0) dim += input.dim();

    // Extract tensor dimensions and compute the sizes for outer, reduction, and inner parts
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

    // The output tensor has the same shape as input except that the reduction dimension becomes 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements (one block per output element)
    int64_t num_output = outer_size * inner_size;

    // Launch configuration
    int threads = 256;  // number of threads per block
    int blocks = num_output;  // one block per output element
    size_t shared_bytes = threads * sizeof(at::ScalarTypeToCPPType(input.scalar_type(), float())); // allocate shared memory per block

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_shared_tile_cuda", ([&] {
        sum_reduce_shared_tile_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_shared_tile_cuda, "Sum reduction using shared memory tile to reduce global memory latency (CUDA)");
}
