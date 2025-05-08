#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>

template <typename scalar_t>
__device__ __forceinline__ const scalar_t* get_slice_ptr(const scalar_t* x, int64_t outer, int64_t inner, int64_t inner_size, int K) {
  return x + outer * (static_cast<int64_t>(K) * inner_size) + inner;
}

template <typename scalar_t>
__device__ __forceinline__ int compute_argmin(const scalar_t* slice_ptr, int K, int64_t stride) {
  scalar_t min_val = slice_ptr[0];
  int min_idx = 0;
  for (int k = 1; k < K; ++k) {
    scalar_t val = slice_ptr[k * stride];
    if (val < min_val) {
      min_val = val;
      min_idx = k;
    }
  }
  return min_idx;
}

template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* __restrict__ x,
                              int64_t* __restrict__ output,
                              int K,
                              int64_t outer_size,
                              int64_t inner_size) {
  const int64_t total_slices = outer_size * inner_size;
  const int64_t start_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_threads = blockDim.x * gridDim.x;
  
  for (int64_t idx = start_idx; idx < total_slices; idx += total_threads) {
    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;
    const scalar_t* slice_ptr = get_slice_ptr(x, outer, inner, inner_size, K);
    output[idx] = compute_argmin(slice_ptr, K, inner_size);
  }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
  int dims = x.dim();
  if (dim < 0) dim += dims;
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) outer_size *= x.size(i);
  int K = static_cast<int>(x.size(dim));
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) inner_size *= x.size(i);

  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) if (i != dim) out_sizes.push_back(x.size(i));
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

  int64_t total_slices = outer_size * inner_size;
  int threads = 256; // Aligning to warp size (32)
  int max_blocks = 2048;
  int blocks = std::min(static_cast<int>((total_slices + threads - 1) / threads), max_blocks);

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    argmin_kernel<scalar_t><<<blocks, threads>>>(
      x.data_ptr<scalar_t>(),
      output.data_ptr<int64_t>(),
      K,
      outer_size,
      inner_size
    );
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA)");
}