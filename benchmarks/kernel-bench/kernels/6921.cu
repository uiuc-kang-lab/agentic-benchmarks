#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// Optimized CUDA kernel for 52_Argmin_over_a_dimension using __ldg() for read-only accesses
// and loop unrolling to improve memory throughput. Assumes that the global memory pointed to by x
// is 128-bit aligned, which is typically the case with PyTorch allocations.

template <typename scalar_t>
__global__ void optimized_argmin_kernel(const scalar_t* __restrict__ x,
                                          int64_t* __restrict__ output,
                                          int K,
                                          int64_t outer_size,
                                          int64_t inner_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total_slices = outer_size * inner_size;
  if (idx >= total_slices) return;

  // Decompose idx into outer and inner indices
  int64_t outer = idx / inner_size;
  int64_t inner = idx % inner_size;

  // Pointer for the current slice. The layout is [outer_size, K, inner_size].
  const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

  // Use __ldg() for read-only loads from global memory
  scalar_t min_val = __ldg(slice_start);
  int min_index = 0;

  // Loop unrolling by factor of 4
  int k = 1;
  for (; k <= K - 4; k += 4) {
    scalar_t val0 = __ldg(slice_start + k * inner_size);
    scalar_t val1 = __ldg(slice_start + (k + 1) * inner_size);
    scalar_t val2 = __ldg(slice_start + (k + 2) * inner_size);
    scalar_t val3 = __ldg(slice_start + (k + 3) * inner_size);
    if (val0 < min_val) { min_val = val0; min_index = k; }
    if (val1 < min_val) { min_val = val1; min_index = k + 1; }
    if (val2 < min_val) { min_val = val2; min_index = k + 2; }
    if (val3 < min_val) { min_val = val3; min_index = k + 3; }
  }
  // Handle the remaining elements
  for (; k < K; ++k) {
    scalar_t val = __ldg(slice_start + k * inner_size);
    if (val < min_val) { min_val = val; min_index = k; }
  }

  // Write the result
  output[idx] = min_index;
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  // Ensure the input tensor is a CUDA tensor
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

  int dims = x.dim();
  if (dim < 0) {
    dim += dims;
  }
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  // Compute outer_size, K (reduction dimension), and inner_size as in the reference implementation
  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  int K = static_cast<int>(x.size(dim));
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }

  // Build output tensor shape (removes the reduction dimension)
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

  // Total number of slices along which we do the reduction
  int64_t total_slices = outer_size * inner_size;
  int threads = 256;
  int blocks = (total_slices + threads - 1) / threads;

  // Dispatch based on the scalar type
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    const scalar_t* x_data = x.data_ptr<scalar_t>();
    int64_t* output_data = output.data_ptr<int64_t>();
    optimized_argmin_kernel<scalar_t><<<blocks, threads>>>(x_data, output_data, K, outer_size, inner_size);
  }));

  // Check for any kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Optimized Argmin forward (CUDA)");
}
