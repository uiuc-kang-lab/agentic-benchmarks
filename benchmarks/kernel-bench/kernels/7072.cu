#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Modular device function: Performs warp-level min reduction using shuffle operations
template <typename scalar_t>
__device__ inline scalar_t warp_reduce_min(scalar_t val) {
  // Assuming warp size of 32
  for (int offset = 16; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, val, offset);
    val = (other < val) ? other : val;
  }
  return val;
}

// Modular device function: Computes the base index for reduction
// Each warp corresponds to one output element by mapping its warp_id to (outer, inner) indices
__device__ inline int get_base_index(int warp_id, int inner, int r) {
  int outer_idx = warp_id / inner;
  int inner_idx = warp_id % inner;
  return outer_idx * (r * inner) + inner_idx;
}

// Modular device function: Each thread in a warp computes a local min over portions of the reduction dimension
template <typename scalar_t>
__device__ inline scalar_t get_local_min(const scalar_t* __restrict__ input, int base, int inner, int r, int lane) {
  scalar_t local_min = std::numeric_limits<scalar_t>::max();
  // Each thread processes elements in the reduction dimension with stride of warp size (32)
  for (int j = lane; j < r; j += 32) {
    int idx = base + j * inner;
    scalar_t val = input[idx];
    local_min = (val < local_min) ? val : local_min;
  }
  return local_min;
}

// Main kernel: Refactored into modular device functions for improved readability and maintainability.
// Each warp (32 threads) computes one output element by reducing over the 'r' dimension of the input tensor.
// The input tensor is logically reshaped as [outer, r, inner].

template <typename scalar_t>
__global__ void modular_min_reduce_kernel(const scalar_t* __restrict__ input,
                                             scalar_t* __restrict__ output,
                                             int outer, int r, int inner) {
  const int warpSize = 32;
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = global_thread_id / warpSize;
  if (warp_id >= outer * inner) return;

  int lane = threadIdx.x % warpSize;
  int base = get_base_index(warp_id, inner, r);
  
  // Each thread computes a local minimum from portions of the reduction dimension
  scalar_t local_min = get_local_min<scalar_t>(input, base, inner, r, lane);
  
  // Use warp-level reduction to combine minima across the warp
  local_min = warp_reduce_min<scalar_t>(local_min);

  // The first lane writes the final minimum to the output
  if (lane == 0) {
    output[warp_id] = local_min;
  }
}

// Forward function: Prepares tensor dimensions, output shape, and launches the kernel
torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor.");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute sizes for outer, reduction (r), and inner dimensions
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Build the output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Each warp (32 threads) computes one output element
  int total_warps = outer * inner;
  const int threads_per_block = 128;  // Tune as needed
  int num_blocks = (total_warps * 32 + threads_per_block - 1) / threads_per_block;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "modular_min_reduce_kernel", ([&] {
    modular_min_reduce_kernel<scalar_t><<<num_blocks, threads_per_block, 0,
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
  m.def("forward", &forward, "Modular min reduction over a specified dimension using device functions (CUDA)");
}
