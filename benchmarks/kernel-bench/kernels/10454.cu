#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses a 2D grid where gridDim.x corresponds to the outer dimension
// and gridDim.y corresponds to the inner dimension. Each block processes one entire
// column of the cumulative sum (along the stride dimension) using a tiled scan
// approach with shared memory. Within each tile, an inclusive scan is performed in
// parallel using a butterfly algorithm. The tile accumulation is carried over to
// subsequent tiles to provide the correct cumulative sum result.

__global__ void cumsum_tile_scan_kernel(const float* __restrict__ input, 
                                          float* output, 
                                          int inner_size, 
                                          int stride) {
    // Determine the outer and inner indices from the 2D grid
    int outer_idx = blockIdx.x; // outer dimension index
    int inner_idx = blockIdx.y; // inner dimension index

    // Each column is of length 'stride'. We use a tiled approach where each tile
    // has a width of blockDim.x elements (tile size).
    
    // Global base offset for this column
    // The layout is: [outer_size, stride, inner_size] with stride along the cumsum dim
    // Global index for element at position s in this column: outer_idx*(stride*inner_size) + s*inner_size + inner_idx

    // Accumulator for the carry from previous tiles
    float accum = 0.0f;

    // Allocate shared memory for tile scan. Each block's shared mem size is declared at launch.
    extern __shared__ float s_data[];

    // Process the column in tiles, each of size blockDim.x
    for (int tile = 0; tile * blockDim.x < stride; tile++) {
        int pos = tile * blockDim.x + threadIdx.x; // position in the column

        // Determine number of valid elements in this tile
        int valid = blockDim.x;
        if ((stride - tile * blockDim.x) < blockDim.x) {
            valid = stride - tile * blockDim.x;
        }

        // Load data from global memory into shared memory if within range
        float val = 0.0f;
        if (pos < stride) {
            int idx = outer_idx * (stride * inner_size) + pos * inner_size + inner_idx;
            val = input[idx];
        }
        s_data[threadIdx.x] = val;
        __syncthreads();

        // Perform an in-place inclusive scan over the tile using shared memory.
        // Only threads with index < valid participate in the scan.
        for (int offset = 1; offset < valid; offset *= 2) {
            float temp = 0.0f;
            if (threadIdx.x >= offset && threadIdx.x < valid) {
                temp = s_data[threadIdx.x - offset];
            }
            __syncthreads();
            if (threadIdx.x < valid) {
                s_data[threadIdx.x] += temp;
            }
            __syncthreads();
        }

        // Add the carry from previous tiles
        if (threadIdx.x < valid && pos < stride) {
            s_data[threadIdx.x] += accum;
        }
        __syncthreads();

        // Update the accumulator for the next tile using the last element in the current tile
        if (threadIdx.x == valid - 1 && pos < stride) {
            accum = s_data[threadIdx.x];
        }
        __syncthreads();

        // Write the results of the current tile back to global memory
        if (pos < stride) {
            int out_idx = outer_idx * (stride * inner_size) + pos * inner_size + inner_idx;
            output[out_idx] = s_data[threadIdx.x];
        }
        __syncthreads();  // Ensure complete writing before next tile iteration
    }
}

// Host function for launching the kernel
// The input tensor x is conceptually reshaped into [outer_size, stride, inner_size]
// where 'stride' is the size along the cumulative dimension (dim). We use a 2D grid:
//   gridDim.x = outer_size, gridDim.y = inner_size
// and each block processes one column along 'stride'.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= x.size(i);
    }

    int stride = x.size(dim);

    // Configure a 2D grid: one block per column (outer x inner)
    dim3 grid(outer_size, inner_size);

    // Choose a tile size, e.g., 256 threads per block (or less if stride is small)
    int tile_size = (stride < 256) ? stride : 256;
    dim3 block(tile_size);

    // Allocate shared memory: one float per thread in the block
    int shared_mem = tile_size * sizeof(float);

    cumsum_tile_scan_kernel<<<grid, block, shared_mem>>>(
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        inner_size, 
        stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with tiled scan and 2D indexing");
}
