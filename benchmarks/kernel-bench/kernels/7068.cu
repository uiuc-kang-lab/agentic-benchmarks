/*
Combined optimized CUDA kernel for min reduction.
This kernel merges the streamlined indexing approach of Kernel 2 with the warp-level reduction and loop unrolling of Kernel 1.
Each warp (32 threads) computes one output element's minimum value over the reduction dimension.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Kernel: each warp handles one output element via warp-level reduction.
// Input is logically reshaped as [outer, r, inner] and reduction is performed along the r dimension.

template <typename scalar_t>
__global__ void min_reduce_warp_unroll_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {

  const int warpSize = 32;
  // Determine warp ID; each warp computes one output element
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (warp_id >= outer * inner) return;

  // Lane within the warp
  int lane = threadIdx.x % warpSize;

  // Map the warp_id to (outer, inner) coordinates
  int outer_idx = warp_id / inner;
  int inner_idx = warp_id % inner;

  // Base pointer for reduction along r dimension
  int base = outer_idx * (r * inner) + inner_idx;

  // Initialize local minimum to maximum possible value
  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  // Each thread processes multiple elements from the r dimension in strides of warpSize
  #pragma unroll
  for (int j = lane; j < r; j += warpSize) {
    int idx = base + j * inner;  // streamlined indexing similar to Kernel 2
    scalar_t curr = input[idx];
    local_min = curr < local_min ? curr : local_min;
  }

  // Warp-level reduction using shuffle down
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      scalar_t tmp = __shfl_down_sync(0xffffffff, local_min, offset);
      local_min = tmp < local_min ? tmp : local_min;
  }

  // The first lane writes the reduced minimum to global memory
  if (lane == 0) {
    output[warp_id] = local_min;
  }
}

// Forward function exposed to Python
torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor.");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate outer, r, and inner dimensions
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Build output shape by excluding the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Each warp of 32 threads computes one output element
  int total_output = outer * inner;
  const int threads_per_block = 128;   // e.g., 128 threads per block
  int num_blocks = (total_output * 32 + threads_per_block - 1) / threads_per_block;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_warp_unroll_cuda", ([&] {
    min_reduce_warp_unroll_kernel<scalar_t><<<num_blocks, threads_per_block, 0,
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
  m.def("forward", &forward, "Combined optimized CUDA kernel for min reduction using warp-level primitives and loop unrolling.");
}
