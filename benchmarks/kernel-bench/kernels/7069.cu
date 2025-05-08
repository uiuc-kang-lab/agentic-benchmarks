/*
Optimized CUDA kernel for min reduction over a specified dimension.
This kernel combines the efficient indexing of Kernel 2 with the warp-level reduction and loop unrolling of Kernel 1.
Each warp (32 threads) computes one output element by iterating over the reduction dimension in strides of the warp size.
Warp-wide reduction is performed using __shfl_down_sync for minimal synchronization overhead.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Combined kernel: each warp computes one output element's min reduction
// The input is logically reshaped as [outer, r, inner] and the reduction is along the r dimension.

template <typename scalar_t>
__global__ void min_reduce_combined_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {

  const int warpSize = 32;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / warpSize; // One warp per output element

  if (warp_id >= outer * inner) return;

  // Lane within the warp
  int lane = thread_id & (warpSize - 1);  // equivalent to thread_id % warpSize

  // Map warp_id to corresponding (outer, inner) coordinates
  int outer_idx = warp_id / inner;
  int inner_idx = warp_id % inner;

  // Compute base pointer for the reduction dimension
  int base = outer_idx * (r * inner) + inner_idx;

  // Initialize local minimum using maximum possible value
  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  // Each thread in a warp iterates over the r dimension in strides of warpSize
  #pragma unroll
  for (int j = lane; j < r; j += warpSize) {
    int idx = base + j * inner;
    scalar_t val = input[idx];
    if (val < local_min) {
      local_min = val;
    }
  }

  // Perform warp-level reduction using shuffle operations
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, local_min, offset);
    if (other < local_min) {
      local_min = other;
    }
  }

  // First lane writes the result
  if (lane == 0) {
    output[warp_id] = local_min;
  }
}

// Forward function: sets up tensor dimensions, computes output shape and launches the kernel
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
  const int threads_per_block = 128;  // e.g., 128 threads per block
  int total_threads = total_output * warpSize;  // one warp per output element
  int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_combined_cuda", ([&] {
    min_reduce_combined_kernel<scalar_t><<<num_blocks, threads_per_block, 0,
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
  m.def("forward", &forward, "Combined min reduction using warp-level primitives and optimized indexing (CUDA)");
}
