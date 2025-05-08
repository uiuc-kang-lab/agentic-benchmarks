#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block tile dimensions
constexpr int TX = 128; // number of threads processing inner dimension
constexpr int TY = 8;   // number of threads parallelizing reduction

// Kernel: Each block processes one outer index and a tile of the inner dimension.
// Threads cooperatively load portions of the reduction dimension into shared memory and reduce them.

template <typename scalar_t>
__global__ void sum_reduce_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {

    // Each block is uniquely identified by two indices: blockIdx.x for outer index and blockIdx.y for inner tile
    int outer_idx = blockIdx.x;
    int inner_tile_start = blockIdx.y * blockDim.x;  // blockDim.x == TX

    // Thread indices within the block
    int tx = threadIdx.x; // index within the inner tile
    int ty = threadIdx.y; // used for parallelizing the reduction

    // Global inner index for this thread
    int inner_idx = inner_tile_start + tx;

    // Allocate shared memory dynamically; size is TX * TY elements
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Each thread computes a partial sum over a subset of the reduction dimension
    scalar_t partial_sum = 0;
    if (inner_idx < inner_size) {
        // Each thread processes a slice of the reduction dimension with stride = blockDim.y (ty dimension)
        for (int r = ty; r < reduce_size; r += blockDim.y) {
            // Compute the index into the flattened input tensor
            // Input layout: [outer_size, reduce_size, inner_size]
            int64_t idx = outer_idx * (reduce_size * inner_size) + r * inner_size + inner_idx;
            partial_sum += input[idx];
        }
    }

    // Store the partial sum into shared memory
    int index = ty * blockDim.x + tx; // Flatten 2D thread index
    sdata[index] = partial_sum;
    __syncthreads();

    // Reduce along the ty dimension (vertical reduction) in shared memory
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        if (ty < offset) {
            sdata[index] += sdata[index + offset * blockDim.x];
        }
        __syncthreads();
    }

    // The first thread in the column (ty == 0) writes the final result
    if (ty == 0 && inner_idx < inner_size) {
        int64_t o_idx = outer_idx * inner_size + inner_idx;
        output[o_idx] = sdata[tx];
    }
}

// CUDA wrapper function
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) {
        dim += input.dim();
    }

    // Extract sizes and compute outer and inner dimensions
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

    // Prepare output tensor by setting the reduced dimension size to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Configure grid dimensions:
    // Each block corresponds to one outer index and a tile of the inner dimension
    dim3 threads(TX, TY);
    dim3 blocks(outer_size, (inner_size + TX - 1) / TX);

    // Calculate shared memory size required per block
    size_t shared_mem_size = TX * TY * sizeof(float);  // placeholder for float; will be replaced per scalar_t

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        sum_reduce_shared_kernel<scalar_t><<<blocks, threads, TX * TY * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA) using shared memory");
}
