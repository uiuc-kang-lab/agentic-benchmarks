#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// This kernel uses __ldg() for read-only global memory loads and assumes that the input is 128-bit aligned.
// The kernel performs a min reduction over the 'r' dimension. Each thread processes one (outer, inner) pair.

template <typename scalar_t>
__global__ void min_reduce_ldg_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (idx >= total) return; // Early exit for out-of-bounds indices

  int outer_idx = idx / inner;
  int inner_idx = idx % inner;
  // Calculate the base index for this (outer, inner) coordinate
  int base = outer_idx * (r * inner) + inner_idx;

  // Use __ldg() for read-only access. Assumes that the pointer is 128-bit aligned.
  scalar_t min_val = __ldg(input + base);

  // Unroll loop in chunks of 4 to reduce loop overhead
  int unroll_end = 1 + ((r - 1) / 4) * 4;
  #pragma unroll 4
  for (int j = 1; j < unroll_end; j += 4) {
    scalar_t val1 = __ldg(input + outer_idx * (r * inner) + (j + 0) * inner + inner_idx);
    scalar_t val2 = __ldg(input + outer_idx * (r * inner) + (j + 1) * inner + inner_idx);
    scalar_t val3 = __ldg(input + outer_idx * (r * inner) + (j + 2) * inner + inner_idx);
    scalar_t val4 = __ldg(input + outer_idx * (r * inner) + (j + 3) * inner + inner_idx);
    min_val = min(min_val, val1);
    min_val = min(min_val, val2);
    min_val = min(min_val, val3);
    min_val = min(min_val, val4);
  }
  
  // Process remaining elements
  for (int j = unroll_end; j < r; j++) {
    scalar_t curr = __ldg(input + outer_idx * (r * inner) + j * inner + inner_idx);
    min_val = min(min_val, curr);
  }

  output[idx] = min_val;
}

// Host function

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute dimensions: outer product of dims before 'dim', r=size of reduced dim, and inner product after 'dim'
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Form output shape by removing the reduced dimension
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

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_ldg_cuda", ([&] {
    min_reduce_ldg_kernel<scalar_t><<<blocks, threads, 0, 
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
  m.def("forward", &forward, "Min reduction using __ldg and aligned memory accesses (CUDA)");
}
