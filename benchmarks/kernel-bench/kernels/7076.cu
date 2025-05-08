#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Define block size as compile-time constant
constexpr int BLOCK_SIZE = 64;  // Experimentally determined optimal value for H100

template <typename scalar_t>
__global__ void min_reduce_adaptive_blocks_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  const int warpSize = 32;
  // Calculate thread indices using smaller block size
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int warp_id = idx / warpSize;
  
  if (warp_id >= outer * inner) return;

  int lane = threadIdx.x % warpSize;
  int outer_idx = warp_id / inner;
  int inner_idx = warp_id % inner;
  int base = outer_idx * (r * inner) + inner_idx;

  // Use registers for temporary storage
  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  // Each thread processes multiple elements with fixed stride
  #pragma unroll 4
  for (int j = lane; j < r; j += warpSize) {
    scalar_t val = input[base + j * inner];
    local_min = (val < local_min) ? val : local_min;
  }

  // Warp-level reduction using shuffle operations
  #pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, local_min, offset);
    local_min = (other < local_min) ? other : local_min;
  }

  if (lane == 0) {
    output[warp_id] = local_min;
  }
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate dimensions
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Create output tensor
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Calculate grid dimensions based on optimized block size
  int total_warps = outer * inner;
  int num_blocks = (total_warps * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_adaptive_blocks", ([&] {
    min_reduce_adaptive_blocks_kernel<scalar_t><<<num_blocks, BLOCK_SIZE, 0,
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
  m.def("forward", &forward, "Min reduction with optimized block size (CUDA)");
}