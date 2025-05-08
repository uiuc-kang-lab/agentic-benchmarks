#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// This kernel assumes the input tensor is logically reshaped as [outer, r, inner],
// where the reduction is performed over the r dimension. The tensor is made contiguous
// so that the innermost dimension (inner) has consecutive memory locations. Each thread
// computes the min reduction for one output element, corresponding to a specific outer and inner index.
// By assigning a 2D grid with grid.x = outer and grid.y covering the inner dimension,
// consecutive threads within a warp load consecutive elements from memory when looping over r,
// ensuring memory coalescing.

template <typename scalar_t>
__global__ void min_reduce_coalesced_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  // Determine the index in the output tensor
  int outer_idx = blockIdx.x;  
  int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

  if (inner_idx >= inner) return;

  // Compute the starting index for the current outer slice in the input
  int base = outer_idx * r * inner;
  scalar_t min_val = std::numeric_limits<scalar_t>::max();

  // Loop over the reduction dimension (r)
  // The access pattern: input[ base + j * inner + inner_idx ]
  // For a fixed j, consecutive threads (varying inner_idx) access consecutive memory locations.
  #pragma unroll
  for (int j = 0; j < r; j++) {
    int index = base + j * inner + inner_idx;
    scalar_t val = input[index];
    if (val < min_val) {
      min_val = val;
    }
  }

  // Write the result to the output tensor which has shape [outer, inner]
  output[outer_idx * inner + inner_idx] = min_val;
}

// Forward function preparing tensor dimensions, output shape and launching the kernel

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute outer: product of dimensions before 'dim', r: size of dimension 'dim',
  // inner: product of dimensions after 'dim'
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

  // Set up 2D grid: grid.x corresponds to the outer dimension; grid.y covers the inner dimension.
  // Using 256 threads per block in x dimension for inner indexing.
  int threads = 256;
  dim3 block(threads);
  dim3 grid(outer, (inner + threads - 1) / threads);

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_coalesced_cuda", ([&] {
    min_reduce_coalesced_kernel<scalar_t><<<grid, block, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      outer,
      r,
      inner);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction over a specified dimension with coalesced memory accesses (CUDA)");
}
