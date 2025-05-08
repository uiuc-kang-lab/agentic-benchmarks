#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// CUDA kernel for optimized min reduction
// Avoids warp divergence by ensuring uniform control flow

template <typename scalar_t>
__global__ void min_reduce_warp_divergence_avoidance_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {

  const int warpSize = 32;
  // Each warp is assigned to compute one output element
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (warp_id >= outer * inner) return;

  // Uniform resource sharing across threads
  int lane = threadIdx.x % warpSize;

  // Calculate the starting index for reduction in the r dimension
  int outer_idx = warp_id / inner;
  int inner_idx = warp_id % inner;
  int base = outer_idx * (r * inner) + inner_idx;

  // Initialize min value
  scalar_t local_min = input[base + lane * inner];

  // Ensure all threads execute the reduction avoiding conditional logic
  for (int j = lane + warpSize; j < r; j += warpSize) {
    int idx = base + j * inner;
    scalar_t val = input[idx];
    local_min = fminf(val, local_min);
  }

  // Uniform warp level reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, local_min, offset);
    local_min = fminf(other, local_min);
  }

  // Store the result in output array
  if (lane == 0) {
    output[warp_id] = local_min;
  }
}

// Forward pass function
// Adjust block and grid configuration

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate dimension sizes
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Create new output tensor with the reduced dimension removed
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Adjust grid to avoid unnecessary thread allocation
  int total_warps = outer * inner;
  const int threads_per_block = 256;  // Efficient occupancy with block size 256
  int num_blocks = (total_warps + threads_per_block / warpSize - 1) / (threads_per_block / warpSize);

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_warp_divergence_avoidance_cuda", ([&] {
    min_reduce_warp_divergence_avoidance_kernel<scalar_t><<<num_blocks, threads_per_block, 0,
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
  m.def("forward", &forward, "Min reduction with warp divergence avoidance (CUDA)");
}
