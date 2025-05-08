#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Optimized CUDA kernel for min reduction over a specified dimension using shared memory.
// Each block handles multiple output elements using shared memory for intra-block reduction.
template <typename scalar_t>
__global__ void min_reduce_shared_memory_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {

  extern __shared__ scalar_t shared_min[];
  const int warpSize = 32;
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = global_thread_id / warpSize;
  if (warp_id >= outer * inner) return;

  int lane = threadIdx.x % warpSize;
  int outer_idx = warp_id / inner;
  int inner_idx = warp_id % inner;
  int base = outer_idx * (r * inner) + inner_idx;

  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  // Load data and compute local min
  for (int j = lane; j < r; j += warpSize) {
    int idx = base + j * inner;
    scalar_t val = input[idx];
    local_min = (val < local_min) ? val : local_min;
  }

  // Store local min into shared memory
  shared_min[threadIdx.x] = local_min;
  __syncthreads();

  // Reduce within the block using shared memory
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    if (lane < offset) {
      shared_min[threadIdx.x] = min(shared_min[threadIdx.x], shared_min[threadIdx.x + offset]);
    }
    __syncthreads();
  }

  // The first thread in each block writes the result
  if (lane == 0) {
    output[warp_id] = shared_min[threadIdx.x];
  }
}

// Forward function: sets up tensor dimensions, output shape and kernel launch parameters
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
  const int threads_per_block = 128;  // 128 threads/block = 4 warps per block
  int num_blocks = (total_output * 32 + threads_per_block - 1) / threads_per_block;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_shared_memory_cuda", ([&] {
    min_reduce_shared_memory_kernel<scalar_t><<<num_blocks, threads_per_block, threads_per_block * sizeof(scalar_t),
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
  m.def("forward", &forward, "Min reduction with shared memory optimization (CUDA)");
}
