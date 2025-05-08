#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// This kernel uses a grid-stride loop over warps to handle a workload
// larger than the available threads. Each warp processes one or more output
// elements by looping over the reduction dimension in strides of 32 (warp size).

template <typename scalar_t>
__global__ void min_reduce_stride_grid_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  const int warpSize = 32;
  // Calculate total number of warps in the grid
  int total_warps = (blockDim.x * gridDim.x) / warpSize;

  // Get global thread id and compute warp id and lane
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = global_thread_id / warpSize;
  int lane = global_thread_id % warpSize;

  // Process output elements (there are outer * inner elements) in a grid-stride loop over warps
  for (int out_idx = warp_id; out_idx < (outer * inner); out_idx += total_warps) {
    // Decode the output index into outer and inner coordinates
    int outer_idx = out_idx / inner;
    int inner_idx = out_idx % inner;

    // Compute the base index for the reduction dimension
    int base = outer_idx * (r * inner) + inner_idx;
    
    // Initialize local minimum with the maximum possible value
    scalar_t local_min = std::numeric_limits<scalar_t>::max();

    // Use a stride loop over the reduction dimension
    for (int j = lane; j < r; j += warpSize) {
      int idx = base + j * inner;
      scalar_t val = input[idx];
      if (val < local_min) {
        local_min = val;
      }
    }

    // Perform warp-level reduction using shuffle down
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      scalar_t tmp = __shfl_down_sync(0xffffffff, local_min, offset);
      if (tmp < local_min) {
        local_min = tmp;
      }
    }

    // The first lane of the warp writes the result
    if (lane == 0) {
      output[out_idx] = local_min;
    }
  }
}

// Forward function for the PyBind11 module

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute outer, r (reduction dim), and inner dimensions
  int outer = 1;
  for (int i = 0; i < dim; ++i) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; ++i) {
    inner *= input.size(i);
  }

  // Create output shape by excluding the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; ++i) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // There are outer * inner output elements. Each warp will process one element.
  // We'll choose a block size (e.g., 128 threads) and calculate grid size accordingly.
  const int threads_per_block = 128;
  int total_output_elements = outer * inner;
  // Each warp consists of 32 threads so total threads needed = total_output_elements * 32
  int total_threads_needed = total_output_elements * warpSize;
  int num_blocks = (total_threads_needed + threads_per_block - 1) / threads_per_block;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_stride_grid_cuda", ([&] {
    min_reduce_stride_grid_kernel<scalar_t><<<num_blocks, threads_per_block, 0,
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
  m.def("forward", &forward, "Min reduction over a specified dimension using grid-stride loops (CUDA)");
}
