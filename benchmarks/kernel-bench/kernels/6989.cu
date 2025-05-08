#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// Optimized CUDA kernel: Combines 2D grid configuration with improved memory coalescing and reduced branching.
template <typename scalar_t>
__global__ void optimized_argmin_kernel(const scalar_t* __restrict__ x,
                                        int64_t* __restrict__ output,
                                        int K,
                                        int64_t outer_size,
                                        int64_t inner_size) {
    // Shared memory for tile processing
    extern __shared__ char shared_mem[];
    scalar_t* tile = reinterpret_cast<scalar_t*>(shared_mem);
    
    // Each block in y corresponds to one outer index
    int outer = blockIdx.y;
    // Compute the inner index from blockIdx.x and threadIdx.x
    int inner = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if(inner >= inner_size) return;
    
    // Data layout is interpreted as: [outer, K, inner]
    const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;
    
    // Initialize with the first element
    scalar_t min_val = slice_start[0];
    int min_index = 0;
    
    // Process K dimension in tiles
    constexpr int TILE_SIZE = 32;  // Adjust based on shared memory size and occupancy requirements
    
    for (int tile_start = 0; tile_start < K; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, K);
        
        // Load tile into shared memory
        for (int k = tile_start + tid; k < tile_end; k += blockDim.x) {
            tile[k - tile_start] = slice_start[k * inner_size];
        }
        __syncthreads();
        
        // Process the tile
        for (int k = 0; k < (tile_end - tile_start); ++k) {
            scalar_t val = tile[k];
            if (val < min_val) {
                min_val = val;
                min_index = tile_start + k;
            }
        }
        __syncthreads();
    }
    
    // Write the result into the output tensor
    output[outer * inner_size + inner] = min_index;
}

at::Tensor argmin_optimized_cuda_forward(const at::Tensor &x, int64_t dim) {
  // Ensure the input is a CUDA tensor.
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

  int dims = x.dim();
  if (dim < 0) {
    dim += dims;
  }
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  // For tensor of shape [d0, d1, ..., d_{D-1}], reshape it conceptually to [outer, K, inner]
  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  int K = static_cast<int>(x.size(dim));
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }

  // The output tensor has the reduction dimension removed; its total number of elements equals outer_size * inner_size.
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

  // Configure kernel launch to use a 2D grid:
  //   - gridDim.y corresponds to outer_size (each slice of the reduction)
  //   - gridDim.x covers the inner dimension, with blockDim.x threads per block
  int threads = 256;
  dim3 block_dim(threads);
  dim3 grid_dim((inner_size + threads - 1) / threads, outer_size);

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_optimized_cuda_forward", ([&] {
    const scalar_t* x_data = x.data_ptr<scalar_t>();
    int64_t* output_data = output.data_ptr<int64_t>();
    optimized_argmin_kernel<scalar_t><<<grid_dim, block_dim>>>(x_data, output_data, K, outer_size, inner_size);
  }));

  // Check for any kernel launch errors.
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_optimized_cuda_forward, "Optimized Argmin forward (CUDA)");
}