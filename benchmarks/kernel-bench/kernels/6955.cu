#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <limits>

// This kernel leverages shared memory by loading tiles of the reduction dimension into shared memory.
// Each block processes one slice of the input tensor along the reduction dimension.
// The kernel loads a tile (of size equal to the blockDim.x) into shared memory, performs an in-tile reduction,
// and then updates the block's running minimum. This approach minimizes repeated global memory accesses.

template <typename scalar_t>
__global__ void argmin_shared_tiled_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {

    // Each block processes one slice (combination of outer and inner indices)
    int slice = blockIdx.x;
    if (slice >= outer_size * inner_size) return;

    int64_t outer = slice / inner_size;
    int64_t inner = slice % inner_size;

    const int tid = threadIdx.x;

    // Declare dynamic shared memory. Layout: first, an array for scalar_t values, then an array for int indices.
    extern __shared__ char smem[];
    scalar_t* tile = reinterpret_cast<scalar_t*>(smem);
    int* tile_idx = reinterpret_cast<int*>(tile + blockDim.x);

    // Initialize the block's running minimum in registers
    scalar_t block_min = std::numeric_limits<scalar_t>::max();
    int block_min_index = 0;

    // Loop over the reduction dimension K in tiles of size blockDim.x
    for (int base = 0; base < K; base += blockDim.x) {
        int k = base + tid;
        if (k < K) {
            // Load one element from global memory using __ldg for cached read-only access
            tile[tid] = __ldg(&x[ outer * (static_cast<int64_t>(K) * inner_size) + k * inner_size + inner ]);
            tile_idx[tid] = k;
        } else {
            tile[tid] = std::numeric_limits<scalar_t>::max();
            tile_idx[tid] = -1;
        }
        __syncthreads();

        // In-tile reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (tile[tid + stride] < tile[tid]) {
                    tile[tid] = tile[tid + stride];
                    tile_idx[tid] = tile_idx[tid + stride];
                }
            }
            __syncthreads();
        }

        // Tile's minimum is now in tile[0]. Thread 0 updates the block's running minimum.
        if (tid == 0) {
            if (tile[0] < block_min) {
                block_min = tile[0];
                block_min_index = tile_idx[0];
            }
        }
        __syncthreads(); // Ensure all threads complete before next tile load
    }

    // Write the final argmin index for this slice to global memory
    if (tid == 0) {
        output[slice] = block_min_index;
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    // Compute outer_size = product of dimensions before 'dim'
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    // K is the size of the reduction dimension
    int K = static_cast<int>(x.size(dim));
    // Compute inner_size = product of dimensions after 'dim'
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) {
        inner_size *= x.size(i);
    }

    // Prepare the output tensor with the reduction dimension removed
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i == dim) continue;
        out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    // Configure kernel launch parameters
    int threads = 128; // number of threads per block (tile size)
    int blocks = outer_size * inner_size; // one block per slice
    size_t shared_mem_size = threads * (sizeof(float) + sizeof(int));
    // Note: 'float' is used here as a placeholder; the actual size depends on scalar_t.

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        size_t smem = threads * (sizeof(scalar_t) + sizeof(int));
        argmin_shared_tiled_kernel<scalar_t><<<blocks, threads, smem>>>(x_data, output_data, K, outer_size, inner_size);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA)");
}
