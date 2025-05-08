#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions for shared memory tiling
#define TILE_R 32
#define TILE_INNER 32

// Kernel using shared memory to stage reduction data
// The input tensor is assumed to have shape [outer, dim, inner] where
// outer = product(sizes[0..dim-1]),
// dim = sizes[dim], and
// inner = product(sizes[dim+1..end]).
// The output tensor has shape [outer, inner] (i.e. the dim dimension is reduced by computing the mean).

template <typename scalar_t>
__global__ void mean_reduce_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t dim_size,
    int64_t inner_size) {
  // Determine the outer index from grid.x
  int outer = blockIdx.x;
  // Determine the starting index for this inner tile from grid.y
  int inner_tile_start = blockIdx.y * TILE_INNER;

  // Thread indices within the block
  int tx = threadIdx.x; // corresponds to a column within the inner tile
  int ty = threadIdx.y; // used for cooperative loading from global memory
  
  // Global inner index for the output element
  int inner_index = inner_tile_start + tx;

  // Allocate shared memory statically for the tile
  __shared__ scalar_t tile[TILE_R * TILE_INNER];

  // Each thread will accumulate the partial sum for one output element
  scalar_t sum = 0;

  // Loop over the reduction dimension in chunks of TILE_R
  for (int tile_start = 0; tile_start < dim_size; tile_start += TILE_R) {
    int i = tile_start + ty;  // reduction index for this thread in y-dimension
    scalar_t val = 0;
    if ((i < dim_size) && (inner_index < inner_size)) {
      // Compute global index in input: 
      // index = outer * (dim_size * inner_size) + i * inner_size + (inner_tile_start + tx)
      int idx = outer * (dim_size * inner_size) + i * inner_size + inner_index;
      val = input[idx];
    } else {
      val = 0;
    }
    
    // Each thread loads one element into shared memory
    tile[ty * TILE_INNER + tx] = val;
    __syncthreads();

    // Only the threads in the first row sum the loaded tile for their column
    if ((ty == 0) && (inner_index < inner_size)) {
      #pragma unroll
      for (int j = 0; j < TILE_R; j++) {
        if (tile_start + j < dim_size) {
          sum += tile[j * TILE_INNER + tx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final result from threads with ty == 0
  if ((ty == 0) && (inner_index < inner_size)) {
    int out_idx = outer * inner_size + inner_index;
    output[out_idx] = sum / static_cast<scalar_t>(dim_size);
  }
}

// Host function invoked from Python
// It computes the outer_size as the product of dimensions before 'dim' and inner_size as the product of dimensions after 'dim'.
// Then, the kernel is launched with a 2D grid where grid.x covers the outer dimension, and grid.y covers inner tiles of size TILE_INNER.

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
  if (dim < 0) dim += input.dim();

  // Get tensor shape as a vector
  auto sizes = input.sizes().vec();
  int64_t dim_size = sizes[dim];

  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= sizes[i];
  }

  int64_t inner_size = 1;
  for (int i = dim + 1; i < sizes.size(); i++) {
    inner_size *= sizes[i];
  }

  // Remove the reduced dimension from the output shape
  std::vector<int64_t> out_sizes = sizes;
  out_sizes.erase(out_sizes.begin() + dim);
  auto output = torch::empty(out_sizes, input.options());

  // Configure grid and block dimensions
  // Grid.x spans the outer dimension; grid.y covers inner tiles of size TILE_INNER
  dim3 grid(outer_size, (inner_size + TILE_INNER - 1) / TILE_INNER);
  // Block dimensions: TILE_INNER threads for inner dimension and TILE_R threads for cooperative loading
  dim3 block(TILE_INNER, TILE_R);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
    mean_reduce_shared_kernel<scalar_t><<<grid, block>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      dim_size,
      inner_size
    );
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mean_reduce_cuda, "Mean reduction using shared memory (CUDA)");
}
