#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// This kernel uses a 2D grid configuration to improve memory coalescing. 
// Each block in the y-dimension processes one 'outer' slice, and the x-dimension threads process contiguous elements along the inner dimension.
// For each (outer, inner) pair, the kernel iterates over the K dimension and finds the index of the minimum element.

template <typename scalar_t>
__global__ void argmin_coalesce_kernel(const scalar_t* __restrict__ x,
                                         int64_t* __restrict__ output,
                                         int K,
                                         int64_t inner_size) {
  // Each block in y corresponds to one outer index.
  int outer = blockIdx.y;
  // Compute the inner index from blockIdx.x and threadIdx.x
  int inner = blockIdx.x * blockDim.x + threadIdx.x;

  if(inner >= inner_size) return;

  // Data layout is interpreted as: [outer, K, inner]
  // Start of the slice for this (outer, inner) location
  const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

  scalar_t min_val = slice_start[0];
  int min_index = 0;
  // Iterate over the K dimension with strides of inner_size so that
  // threads accessing the same k in different lanes will access contiguous memory.
  for (int k = 1; k < K; ++k) {
    scalar_t val = slice_start[k * inner_size];
    if (val < min_val) {
      min_val = val;
      min_index = k;
    }
  }

  // Write the result into the output tensor. The output is viewed as a 2D array [outer, inner].
  output[outer * inner_size + inner] = min_index;
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  // Ensure the input is a CUDA tensor.
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

  int dims = x.dim();
  if (dim < 0) {
    dim += dims;
  }
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  // For tensor of shape [d0, d1, ..., d_{D-1}], reshape it conceptually to [outer, K, inner]
  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  int K = static_cast<int>(x.size(dim));
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }

  // The output tensor has the reduction dimension removed; its total number of elements equals outer_size * inner_size.
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

  // Configure kernel launch to use a 2D grid:
  //   - gridDim.y corresponds to outer_size (each slice of the reduction)
  //   - gridDim.x covers the inner dimension, with blockDim.x threads per block
  int threads = 256;
  dim3 block_dim(threads);
  dim3 grid_dim((inner_size + threads - 1) / threads, outer_size);

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    const scalar_t* x_data = x.data_ptr<scalar_t>();
    int64_t* output_data = output.data_ptr<int64_t>();
    argmin_coalesce_kernel<scalar_t><<<grid_dim, block_dim>>>(x_data, output_data, K, inner_size);
  }));

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Argmin forward with coalesced memory accesses (CUDA)");
}
