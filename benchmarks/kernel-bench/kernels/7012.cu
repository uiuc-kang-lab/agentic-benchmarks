#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// This kernel leverages shared memory tiling. Each block handles a single outer index and a tile of inner indices.
// For a given outer index, the tensor data is of shape [K, inner_size]. Threads in a block cooperatively load a tile of
// columns (inner indices) into shared memory and perform the reduction (argmin) along the K dimension.

template <typename scalar_t>
__global__ void argmin_tile_kernel(const scalar_t* __restrict__ x,
                                     int64_t* __restrict__ output,
                                     int K,
                                     int inner_size,
                                     int64_t outer_size) {
    // Each block in the x-dimension corresponds to one outer index
    int outer = blockIdx.x;
    // Each block in the y-dimension processes a tile of the inner dimension
    int inner_base = blockIdx.y * blockDim.x;
    int tid = threadIdx.x;
    int inner = inner_base + tid;

    // Allocate shared memory dynamically for the tile: size = K * blockDim.x
    extern __shared__ char shared_mem[];
    scalar_t* tile = reinterpret_cast<scalar_t*>(shared_mem);

    // Load the K elements for this tile column from global memory into shared memory
    // Data layout: [outer, K, inner_size] such that element = x[outer * (K * inner_size) + k * inner_size + inner]
    for (int k = 0; k < K; ++k) {
        if (inner < inner_size)
            tile[k * blockDim.x + tid] = x[outer * (static_cast<int64_t>(K) * inner_size) + k * inner_size + inner];
        else
            tile[k * blockDim.x + tid] = scalar_t(0);  // Dummy value for threads out of bounds
    }
    __syncthreads();

    // Only threads with a valid inner index perform the reduction
    if (inner < inner_size) {
        scalar_t min_val = tile[0 * blockDim.x + tid];
        int argmin = 0;
        for (int k = 1; k < K; k++) {
            scalar_t curr = tile[k * blockDim.x + tid];
            if (curr < min_val) {
                min_val = curr;
                argmin = k;
            }
        }
        // Write the result to the output tensor with shape [outer_size, inner_size]
        output[outer * inner_size + inner] = argmin;
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    int dims = x.dim();
    if (dim < 0) {
        dim += dims;
    }
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    // Compute the sizes for outer, reduction (K), and inner dimensions
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    int K = static_cast<int>(x.size(dim));
    int inner_size = 1;
    for (int i = dim + 1; i < dims; i++) {
        inner_size *= x.size(i);
    }

    // The output tensor has shape [outer_size, inner_size] (i.e. original shape without the reduced dimension)
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i == dim) continue;
        out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    // Define tiling parameters. Each block processes a tile of the inner dimension.
    constexpr int tile_width = 256;
    dim3 threads(tile_width);
    dim3 blocks(outer_size, (inner_size + tile_width - 1) / tile_width);

    // Launch the kernel with shared memory size = tile_width * K * sizeof(scalar_t)
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        size_t shared_mem_size = tile_width * K * sizeof(scalar_t);
        argmin_tile_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            x_data,
            output_data,
            K,
            inner_size,
            outer_size
        );
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
