#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// Optimized CUDA kernel for performing a min reduction along a specified dimension.
// Combines warp-level reduction with 2D grid mapping for better efficiency and occupancy.

template <typename scalar_t>
__global__ void optimized_min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
  // 2D grid mapping for outer and inner dimensions
  int outer_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer_idx >= outer || inner_idx >= inner) return;
  
  int base = outer_idx * (r * inner) + inner_idx;
  scalar_t my_min = std::numeric_limits<scalar_t>::max();

  // Each thread computes a partial min over the reduction dimension with stride = blockDim.x
  for (int j = threadIdx.x; j < r; j += blockDim.x) {
    int pos = base + j * inner;
    scalar_t val = input[pos];
    if (val < my_min) {
      my_min = val;
    }
  }

  // Use shared memory for block-level reduction
  extern __shared__ scalar_t shared_min[];
  shared_min[threadIdx.x] = my_min;
  __syncthreads();

  // Perform reduction within the block
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared_min[threadIdx.x] = min(shared_min[threadIdx.x], shared_min[threadIdx.x + offset]);
    }
    __syncthreads();
  }

  // The first thread in the block writes the result
  if (threadIdx.x == 0) {
    int out_index = outer_idx * inner + inner_idx;
    output[out_index] = shared_min[0];
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

  // Calculate sizes: 'outer' are dimensions before reduced dimension,
  // 'r' is the size of the reduced dimension, and 'inner' are dimensions after.
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Create output shape by removing the reduced dimension.
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  
  // Allocate output tensor.
  auto output = torch::empty(output_shape, input.options());

  // Define 2D grid and block dimensions for the outer and inner dimensions.
  const dim3 threads(32, 8);  // 256 threads per block
  const dim3 blocks((inner + threads.x - 1) / threads.x, (outer + threads.y - 1) / threads.y);

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "optimized_min_reduce_cuda", ([&] {
    optimized_min_reduce_kernel<scalar_t><<<blocks, threads, threads.x * sizeof(scalar_t), 
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
  m.def("forward", &forward, "Optimized min reduction over a specified dimension (CUDA)");
}
