/*
 * This kernel combines the tiled grid structure from kernel1 with the
 * block-level parallel scan and aligned memory accesses (__ldg) from kernel2.
 * It processes multiple cumsum lines (each corresponding to a combination of outer and
 * inner indices) per block using a 2D thread block. The x-dimension of the block
 * partitions the stride dimension (the cumulative sum dimension) among threads,
 * and the y-dimension lets a block process several inner indices (a tile) concurrently.
 * Each thread computes a partial sum over its assigned chunk, then a per-line shared memory
 * inclusive scan is used to compute offsets for the final cumulative sum. This approach
 * improves occupancy and memory coalescing while reducing kernel launch overhead for
 * high inner_size tensors.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Parameters: partition the work along the stride dimension and tile the inner dimension.
#define THREADS_PER_LINE 256
#define TILE_LINES 4

// Combined kernel: each block processes TILE_LINES cumsum lines (each corresponding to an inner index).
// Outer dimension is mapped to grid.x, and the inner dimension is tiled along grid.y.
// The stride dimension is scanned in parallel using a shared memory scan per tile line.

__global__ void tile_scan_cumsum_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int outer_size,
                                          int inner_size,
                                          int stride) {
    // Map blockIdx.x to outer index, and blockIdx.y to a tile (chunk) of inner indices.
    int outer_idx = blockIdx.x;
    int inner_tile_start = blockIdx.y * TILE_LINES;

    // Each block processes TILE_LINES lines; threadIdx.y selects the line within the tile.
    int tile_line = threadIdx.y;
    int global_inner = inner_tile_start + tile_line;

    // If the computed inner index is outside the valid range, exit early.
    if (global_inner >= inner_size)
        return;

    // Compute base pointers for the current line. The cumsum operation runs along the 'stride' dimension.
    // The memory layout follows: [outer][stride][inner]
    const float* in_line = input + outer_idx * stride * inner_size + global_inner;
    float* out_line = output + outer_idx * stride * inner_size + global_inner;

    // Divide the cumulative sum (stride) among threads along the x-dimension.
    int chunk_size = (stride + THREADS_PER_LINE - 1) / THREADS_PER_LINE;
    int start = threadIdx.x * chunk_size;
    int end = min(start + chunk_size, stride);

    // First pass: each thread computes the partial sum of its assigned chunk.
    float thread_sum = 0.0f;
    for (int i = start; i < end; i++) {
        // Stride between consecutive elements is inner_size.
        thread_sum += __ldg(&in_line[i * inner_size]);
    }

    // Allocate shared memory for performing an inclusive scan for each tile line separately.
    // Shared memory layout: each tile line gets THREADS_PER_LINE floats consecutively.
    extern __shared__ float sdata[];
    int smem_index = tile_line * THREADS_PER_LINE + threadIdx.x;
    sdata[smem_index] = thread_sum;
    __syncthreads();

    // Perform an inclusive scan on the partial sums for each line.
    for (int offset = 1; offset < THREADS_PER_LINE; offset *= 2) {
        float temp = 0.0f;
        if (threadIdx.x >= offset) {
            temp = sdata[tile_line * THREADS_PER_LINE + threadIdx.x - offset];
        }
        __syncthreads();
        sdata[smem_index] += temp;
        __syncthreads();
    }

    // The offset for a thread's chunk is the sum of all previous threads' partial sums.
    float add_offset = (threadIdx.x == 0) ? 0.0f : sdata[tile_line * THREADS_PER_LINE + threadIdx.x - 1];

    // Second pass: each thread recomputes its local cumulative sum and writes the final results
    // by adding the computed offset.
    float local_running = 0.0f;
    for (int i = start; i < end; i++) {
        local_running += __ldg(&in_line[i * inner_size]);
        out_line[i * inner_size] = local_running + add_offset;
    }
}

// The forward function sets up the tensor dimensions and grid/block configuration.
// It maps the outer dimensions to grid.x and tiles the inner dimensions over grid.y.
// Shared memory allocation is based on TILE_LINES * THREADS_PER_LINE floats.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;  // support negative dims

    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= x.size(i);
    }

    int stride = x.size(dim);

    // Grid: one block per outer index and per tile of the inner dimension.
    dim3 grid(outer_size, (inner_size + TILE_LINES - 1) / TILE_LINES);
    // Block: THREADS_PER_LINE threads for cumsum along stride, and TILE_LINES lines per block.
    dim3 block(THREADS_PER_LINE, TILE_LINES);

    size_t sharedMemBytes = TILE_LINES * THREADS_PER_LINE * sizeof(float);

    tile_scan_cumsum_kernel<<<grid, block, sharedMemBytes>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        outer_size,
        inner_size,
        stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with tiled multi-line block scan, __ldg optimization, and improved occupancy");
}
