#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Optimized kernel with block size tuning for min reduction
// Each warp computes the min reduction for one output element using warp-level primitives
// The input is logically reshaped as [outer, r, inner], and the reduction is performed along the r dimension.

// Kernel function
template <typename scalar_t>
__global__ void min_reduce_block_size_tuning_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {

  const int warpSize = 32;
  // Each warp is assigned one output element
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (warp_id >= outer * inner) return;

  // Determine lane within the warp
  int lane = threadIdx.x % warpSize;

  // Map warp_id to the corresponding (outer, inner) coordinates
  int outer_idx = warp_id / inner;
  int inner_idx = warp_id % inner;
  // Compute the base pointer offset for the reduction dimension
  int base = outer_idx * (r * inner) + inner_idx;

  // Initialize with maximum possible value
  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  // Each thread in the warp iterates over the reduction dimension in stride of warpSize
  #pragma unroll
  for (int j = lane; j < r; j += warpSize) {
    int idx = base + j * inner;
    scalar_t val = input[idx];
    local_min = (val < local_min) ? val : local_min;
  }

  // Warp-level reduction using shuffle operations
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, local_min, offset);
    local_min = (other < local_min) ? other : local_min;
  }

  // The first lane writes the result
  if (lane == 0) {
    output[warp_id] = local_min;
  }
}

// Forward function: sets up tensor dimensions, output shape and kernel launch parameters
torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute sizes: outer dimensions, reduction dimension (r), and inner dimensions
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Build output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Each warp (32 threads) computes one output element
  int total_output = outer * inner;
  const int threads_per_block = 256;  // Experimenting with 256 threads/block
  int num_blocks = (total_output * 32 + threads_per_block - 1) / threads_per_block;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_block_size_tuning_cuda", ([&] {
    min_reduce_block_size_tuning_kernel<scalar_t><<<num_blocks, threads_per_block, 0,
      c10::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        outer,
        r,
        inner);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction with block size tuning (CUDA)");
}
