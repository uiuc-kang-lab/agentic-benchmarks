#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// This kernel uses warp-level primitives to perform a min reduction along the specified dimension.
// It assumes that the reduction dimension (r) is small (r <= 32) so that one warp can cover all required elements.

template <typename scalar_t>
__global__ void warp_min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  // Each warp processes one output element.
  // Compute global thread id, warp id, and lane (thread index within the warp)
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / warpSize;  // warpSize is 32
  int lane = thread_id % warpSize;

  int total_groups = outer * inner;
  if (warp_id >= total_groups) return;

  // Compute the mapping from warp_id to the corresponding (outer, inner) indices
  int outer_idx = warp_id / inner;
  int inner_idx = warp_id % inner;

  // Base index in the input tensor for this reduction group
  int base = outer_idx * (r * inner) + inner_idx;

  // Initialize with the first element
  scalar_t val = std::numeric_limits<scalar_t>::max();
  if (lane < r) {
    val = input[base + lane * inner];
  }

  // Create mask for active threads based on reduction size
  unsigned mask = (1U << r) - 1;  // Only include threads that have valid data
  
  // Perform warp-level reduction using __shfl_down_sync
  // Use mask to only include threads with valid data
  for (int offset = 1; offset < r; offset *= 2) {
    scalar_t other = __shfl_down_sync(mask, val, offset);
    if (lane + offset < r) {  // Only update if the other value is from a valid position
      val = min(val, other);
    }
  }

  // Lane 0 in each warp writes the result
  if (lane == 0) {
    output[warp_id] = val;
  }
}


// The forward function takes a CUDA tensor and a dimension along which to reduce
// It arranges the tensor into [outer, r, inner] shape logically, similar to the reference implementation.
// Then, each warp performs the min reduction over the 'r' dimension using warp shuffles.

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute 'outer' (product of dims before 'dim'), 'r' (size of the reduction dim), and 'inner' (product of dims after 'dim')
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Create output shape by removing the reduction dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Each reduction group (of size 'r') is handled by one warp
  int total_groups = outer * inner;
  int total_threads = total_groups * 32; // Launch one warp (32 threads) per group
  const int threads = 256;
  const int blocks = (total_threads + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "warp_min_reduce_cuda", ([&] {
    warp_min_reduce_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      outer,
      r,
      inner);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction over a specified dimension (CUDA) using warp-level primitives");
}
