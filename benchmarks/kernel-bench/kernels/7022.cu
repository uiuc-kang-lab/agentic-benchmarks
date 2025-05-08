#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// Helper device function to compute the pointer to the start of a slice.
// We assume that the input tensor is allocated with proper alignment for 128-bit accesses.
template <typename scalar_t>
__device__ __forceinline__ const scalar_t* get_slice_ptr(const scalar_t* __restrict__ x,
                                                            int64_t outer,
                                                            int64_t inner,
                                                            int64_t inner_size,
                                                            int K) {
  // Since tensors from PyTorch are typically allocated with 128-bit alignment,
  // this pointer arithmetic will result in pointers aligned to 128 bits when possible.
  return x + outer * (static_cast<int64_t>(K) * inner_size) + inner;
}

// Helper device function to compute argmin along a slice using __ldg() to leverage the read-only cache.
// The slice is assumed to have K elements with a stride of 'stride' between them.
template <typename scalar_t>
__device__ __forceinline__ int compute_argmin(const scalar_t* __restrict__ slice_ptr, int K, int64_t stride) {
  // Use __ldg() for read-only global memory access, which can improve performance when the data is cached.
  scalar_t min_val = __ldg(&slice_ptr[0]);
  int min_idx = 0;
  #pragma unroll
  for (int k = 1; k < K; ++k) {
    scalar_t val = __ldg(&slice_ptr[k * stride]);
    if (val < min_val) {
      min_val = val;
      min_idx = k;
    }
  }
  return min_idx;
}

// Kernel that computes the argmin for each slice along the specified dimension
// using the helper functions with __ldg() and assumes 128-bit aligned memory accesses.
template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* __restrict__ x,
                                int64_t* __restrict__ output,
                                int K,
                                int64_t outer_size,
                                int64_t inner_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total_slices = outer_size * inner_size;
  if (idx >= total_slices) return;

  int64_t outer = idx / inner_size;
  int64_t inner = idx % inner_size;

  const scalar_t* slice_ptr = get_slice_ptr(x, outer, inner, inner_size, K);
  int min_idx = compute_argmin(slice_ptr, K, inner_size);
  output[idx] = min_idx;
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
  int dims = x.dim();
  if (dim < 0) {
    dim += dims;
  }
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  // Compute outer_size, K, inner_size based on the reduction dimension
  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  int K = static_cast<int>(x.size(dim));
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }

  // Prepare the output tensor shape (removing the reduction dimension)
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

  int64_t total_slices = outer_size * inner_size;
  int threads = 256;
  int blocks = (total_slices + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    const scalar_t* x_data = x.data_ptr<scalar_t>();
    int64_t* output_data = output.data_ptr<int64_t>();
    argmin_kernel<scalar_t><<<blocks, threads>>>(x_data, output_data, K, outer_size, inner_size);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA)");
}
