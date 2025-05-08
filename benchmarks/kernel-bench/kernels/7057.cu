#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Define a default block size which can be tuned (e.g., 32, 64, 128, 256, 512)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Each warp (32 threads) computes the min reduction for one output element
// The input tensor is logically reshaped as [outer, r, inner] and the reduction is performed along the r dimension

template <typename scalar_t>
__global__ void min_reduce_tunable_blocksize_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {
  const int warpSize = 32;
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = global_thread_id / warpSize;
  if (warp_id >= outer * inner) return;

  int lane = threadIdx.x % warpSize;
  int outer_idx = warp_id / inner;
  int inner_idx = warp_id % inner;
  int base = outer_idx * (r * inner) + inner_idx;

  // Initialize with maximum value
  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  // Loop over the reduction dimension in strides of warpSize
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

  // The first lane in the warp writes the result
  if (lane == 0) {
    output[warp_id] = local_min;
  }
}

// Forward function: prepares tensor dimensions, output shape, and launches the kernel
// An additional parameter 'block_size' allows runtime tuning of the block configuration

torch::Tensor forward(torch::Tensor input, int64_t dim, int block_size = BLOCK_SIZE) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate sizes: outer dimensions, reduction dimension (r), and inner dimensions
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
  int total_warps = outer * inner;
  int total_threads = total_warps * 32;
  int num_blocks = (total_threads + block_size - 1) / block_size;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_tunable_blocksize_cuda", ([&] {
    min_reduce_tunable_blocksize_kernel<scalar_t><<<num_blocks, block_size, 0, 
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
  m.def("forward", &forward, "Min reduction over a specified dimension with tunable block size (CUDA)",
        pybind11::arg("input"), pybind11::arg("dim"), pybind11::arg("block_size") = BLOCK_SIZE);
}
