/*
Optimized CUDA kernel for min reduction combining loop unrolling and streamlined indexing.
This kernel logically reshapes the input as [outer, r, inner] and performs reduction along the r dimension.
Each warp (32 threads) computes one output element via warp-level reduction using __shfl_down_sync.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Helper function to perform warp-level reduction of min
template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceMin(scalar_t val) {
  // The warp size is assumed to be 32
  for (int offset = 16; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, val, offset);
    if (other < val)
      val = other;
  }
  return val;
}

// Combined kernel: each warp computes one output element's min reduction
template <typename scalar_t>
__global__ void min_reduce_combined_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {

  const int warpSize = 32;
  // Compute global warp id; each warp of 32 threads processes one output element
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (warp_id >= outer * inner) return;

  int lane = threadIdx.x % warpSize;  // lane index within the warp

  // Map warp_id to corresponding (outer, inner) coordinates
  int outer_idx = warp_id / inner;
  int inner_idx = warp_id % inner;
  int base = outer_idx * (r * inner) + inner_idx;

  // Initialize local minimum using maximum possible value
  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  // Loop over reduction dimension with stride = warpSize and use loop unrolling
  #pragma unroll
  for (int j = lane; j < r; j += warpSize) {
    int idx = base + j * inner;
    scalar_t val = input[idx];
    local_min = (val < local_min) ? val : local_min;
  }

  // Perform warp-wide reduction using shuffle operations
  local_min = warpReduceMin<scalar_t>(local_min);

  // The first lane in the warp writes the result
  if (lane == 0) {
    output[warp_id] = local_min;
  }
}

// Forward function: sets up dimensions, creates output tensor, and launches the kernel
torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate sizes: combine dimensions before and after the reduced dimension
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

  // Each warp (32 threads) computes one element of the output
  int total_warps = outer * inner;
  const int threads_per_block = 128; // e.g., 128 threads per block (4 warps per block)
  int num_blocks = (total_warps * 32 + threads_per_block - 1) / threads_per_block;

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
  m.def("forward", &forward, "Combined optimized min reduction using warp-level primitives (CUDA)");
}
