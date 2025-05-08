#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// Optimized CUDA kernel for min reduction over a specified dimension using loop unrolling.
// The input tensor is logically viewed as [outer, r, inner] where r is the size
// of the reduction dimension. Each thread computes the minimum over the r dimension
// for one (outer, inner) index pair.

template <typename scalar_t>
__global__ void min_reduce_opt_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (idx >= total) return;

  int outer_idx = idx / inner;
  int inner_idx = idx % inner;

  // Compute the base offset for this (outer, inner) position
  int base_offset = outer_idx * (r * inner) + inner_idx;

  // Initialize the minimum value with the first element in the reduction dimension
  scalar_t min_val = input[base_offset];

  // Start reducing from the second element
  int j = 1;
  // Unroll the loop in chunks of 4
  int limit = (r - 1) - ((r - 1) % 4);
  for (; j < limit; j += 4) {
    scalar_t a = input[base_offset + j * inner];
    scalar_t b = input[base_offset + (j + 1) * inner];
    scalar_t c = input[base_offset + (j + 2) * inner];
    scalar_t d = input[base_offset + (j + 3) * inner];

    // Unrolled min comparisons
    min_val = min(min_val, a);
    min_val = min(min_val, b);
    min_val = min(min_val, c);
    min_val = min(min_val, d);
  }

  // Process any remaining elements
  for (; j < r; j++) {
    scalar_t curr = input[base_offset + j * inner];
    min_val = min(min_val, curr);
  }

  // Write the minimum value to output
  output[idx] = min_val;
}

// Host function to setup and launch the CUDA kernel

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute dimensions: outer (product of dimensions before dim), r (size of dim), inner (product after dim)
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Build the output shape (removing the reduced dimension)
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  int total = outer * inner;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_cuda", ([&] {
    min_reduce_opt_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
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
