#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define block (tile) size for processing the cumulative dimension in tiles
const int BLOCK_SIZE = 256;

// This kernel performs cumulative sum along a given dimension by processing the "line" in tiles.
// Each block processes one line (combination of outer and inner indices) of the tensor.
// The stride (length of the cumulative sum dimension) is processed in tiles of size BLOCK_SIZE.
// For each tile, values are loaded into shared memory, an inclusive scan is performed, and then
// an offset from previous tiles is added to compute the correct cumulative sum results.
// Stride loops are used to iterate over tiles if the stride is larger than BLOCK_SIZE, with careful
// boundary handling to ensure correctness.

__global__ void cumsum_stride_kernel(const float* input, float* output, int stride, int inner_size) {
    // Each block processes one cumulative sum line
    int line = blockIdx.x; // line index corresponds to a combination of outer and inner indices
    int outer = line / inner_size;
    int inner = line % inner_size;

    // Compute pointers for the current line. Consecutive elements along the cumsum dimension
    // are separated by inner_size in memory.
    const float* in_line = input + outer * stride * inner_size + inner;
    float* out_line = output + outer * stride * inner_size + inner;

    // Shared variable to hold the running offset from previous tiles in this line
    __shared__ float block_offset;
    if (threadIdx.x == 0) {
        block_offset = 0.0f;
    }
    __syncthreads();

    // Shared memory array used for performing the in-tile inclusive scan
    __shared__ float tile_shared[BLOCK_SIZE];

    // Process the stride dimension in tiles of BLOCK_SIZE using a stride loop
    for (int tile_start = 0; tile_start < stride; tile_start += BLOCK_SIZE) {
        int valid = (stride - tile_start) < BLOCK_SIZE ? (stride - tile_start) : BLOCK_SIZE;
        float val = (threadIdx.x < valid) ? in_line[(tile_start + threadIdx.x) * inner_size] : 0.0f;
        if (threadIdx.x < valid) {
            tile_shared[threadIdx.x] = val;
        }
        __syncthreads();

        // In-tile inclusive scan optimized over only valid elements
        for (int offset = 1; offset < valid; offset *= 2) {
            float temp = 0.0f;
            if (threadIdx.x >= offset && threadIdx.x < valid) {
                temp = tile_shared[threadIdx.x - offset];
            }
            __syncthreads();
            if (threadIdx.x < valid) {
                tile_shared[threadIdx.x] += temp;
            }
            __syncthreads();
        }

        if (threadIdx.x < valid) {
            out_line[(tile_start + threadIdx.x) * inner_size] = tile_shared[threadIdx.x] + block_offset;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            block_offset += tile_shared[valid - 1];
        }
        __syncthreads();
    }
}

// The forward function sets up the grid and launches the kernel.
// It computes outer_size and inner_size based on the specified dimension for cumulative sum.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= x.size(i);
    }

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= x.size(i);
    }

    int stride = x.size(dim);
    // Total number of lines to process: one for each combination of outer and inner indices
    int total_lines = outer_size * inner_size;

    // Launch one block per line, using BLOCK_SIZE threads per block
    cumsum_stride_kernel<<<total_lines, BLOCK_SIZE>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), stride, inner_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum using stride loops");
}
