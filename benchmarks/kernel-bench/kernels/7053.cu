#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Improved CUDA kernel that uses warp-level primitives to perform
// a more efficient min reduction along a specified dimension.
template <typename scalar_t>
__global__ void efficient_min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  // Calculate globally unique thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = outer * inner;

  // Each warp is responsible for reducing elements over the r dimension
  int warpId = idx / 32;
  if (warpId >= total_threads) return;

  int outer_idx = warpId / inner;
  int inner_idx = warpId % inner;

  // Starting index for reduction in the r dimension
  int base = outer_idx * (r * inner) + inner_idx;
  int lane = threadIdx.x % 32;

  // Use warp shuffle to calculate minimum
  scalar_t my_min = std::numeric_limits<scalar_t>::max();
  for (int j = lane; j < r; j += 32) {
    scalar_t val = input[base + j * inner];
    if (val < my_min) {
      my_min = val;
    }
  }

  // Perform warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, my_min, offset);
    if (other < my_min) {
      my_min = other;
    }
  }

  // Write reduced value to output array
  if (lane == 0) {
    output[warpId] = my_min;
  }
}

// Forward function: prepares tensor dimensions and launches the kernel

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate sizes
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

  int total_warps = outer * inner;
  int threads_per_block = 128;
  int num_blocks = (total_warps * 32 + threads_per_block - 1) / threads_per_block;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "efficient_min_reduce", ([&] {
    efficient_min_reduce_kernel<scalar_t><<<num_blocks, threads_per_block, 0,
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
  m.def("forward", &forward, "Min reduction over a specified dimension using improved CUDA kernel");
}