#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// CUDA kernel for performing a min reduction along a specified dimension using 2D grid mapping.
// The input is logically reshaped as [outer, r, inner] where 'r' is the size of the reduction dimension.
// Using 2D grid indexing (outer and inner dimensions) can lead to better occupancy and more efficient thread mapping.

template <typename scalar_t>
__global__ void min_reduce_2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
  int outer_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer_idx >= outer || inner_idx >= inner) return;
  
  int base = outer_idx * (r * inner) + inner_idx;
  scalar_t min_val = input[base];
  for (int j = 1; j < r; j++) {
    int index = outer_idx * (r * inner) + j * inner + inner_idx;
    scalar_t curr = input[index];
    if (curr < min_val) {
      min_val = curr;
    }
  }
  int out_index = outer_idx * inner + inner_idx;
  output[out_index] = min_val;
}


// The forward function prepares the tensor dimensions and launches the kernel with a 2D grid.

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
  const dim3 threads(16, 16);
  const dim3 blocks((inner + threads.x - 1) / threads.x, (outer + threads.y - 1) / threads.y);

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_cuda_2d", ([&] {
    min_reduce_2d_kernel<scalar_t><<<blocks, threads, 0, 
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
  m.def("forward", &forward, "Min reduction over a specified dimension (CUDA) with 2D grid indexing");
}
