#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__device__ __forceinline__ const scalar_t* get_slice_ptr(const scalar_t* x,
                                                            int64_t outer,
                                                            int64_t inner,
                                                            int64_t inner_size,
                                                            int K) {
  return x + outer * (static_cast<int64_t>(K) * inner_size) + inner;
}

template <typename scalar_t, int block_size>
__global__ void argmin_kernel(const scalar_t* __restrict__ x,
                              int64_t* __restrict__ output,
                              int K,
                              int64_t outer_size,
                              int64_t inner_size) {
  __shared__ scalar_t shared_vals[block_size];
  __shared__ int shared_idxs[block_size];

  int tid = threadIdx.x;
  int slice_idx = blockIdx.x;

  int64_t outer = slice_idx / inner_size;
  int64_t inner = slice_idx % inner_size;

  const scalar_t* slice_ptr = get_slice_ptr(x, outer, inner, inner_size, K);

  scalar_t local_min_val = INFINITY;
  int local_min_idx = -1;

  // Coalesced memory access pattern
  for (int j = 0; j < (K + block_size - 1) / block_size; ++j) {
    int k = j * block_size + tid;
    if (k < K) {
      scalar_t val = slice_ptr[k * inner_size];
      if (val < local_min_val) {
        local_min_val = val;
        local_min_idx = k;
      }
    }
  }

  // Shared memory reduction
  shared_vals[tid] = local_min_val;
  shared_idxs[tid] = local_min_idx;
  __syncthreads();

  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (shared_vals[tid + s] < shared_vals[tid]) {
        shared_vals[tid] = shared_vals[tid + s];
        shared_idxs[tid] = shared_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[slice_idx] = shared_idxs[0];
  }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
  int dims = x.dim();
  if (dim < 0) dim += dims;
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  int64_t outer_size = 1;
  for (int i = 0; i < dim; ++i) outer_size *= x.size(i);
  int K = x.size(dim);
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; ++i) inner_size *= x.size(i);

  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; ++i) if (i != dim) out_sizes.push_back(x.size(i));
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

  constexpr int block_size = 256;
  int64_t total_slices = outer_size * inner_size;
  int blocks = total_slices;

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    argmin_kernel<scalar_t, block_size><<<blocks, block_size>>>(
      x.data_ptr<scalar_t>(),
      output.data_ptr<int64_t>(),
      K,
      outer_size,
      inner_size
    );
  }));

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA)");
}