#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// 2D kernel with memory coalescing: each block row works on one outer index
// and threads in block.x cover the inner dimension consecutively.

template <typename scalar_t>
__global__ void argmin_2d_kernel(const scalar_t* __restrict__ x,
                                   int64_t* __restrict__ output,
                                   int K,
                                   int64_t inner_size) {
  // Compute inner index from block.x and threadIdx.x
  int inner = blockIdx.x * blockDim.x + threadIdx.x;
  // Each block row corresponds to one outer index
  int outer = blockIdx.y;
  if (inner >= inner_size)
    return;

  // Compute pointer to the start of the slice for the current (outer, inner).
  // The input tensor is viewed as [outer_size, K, inner_size] in row-major order.
  const scalar_t* slice_ptr = x + static_cast<int64_t>(outer) * (K * inner_size) + inner;

  // Initialize the minimum value and index using the first element of the slice.
  scalar_t min_val = __ldg(&slice_ptr[0]);
  int min_idx = 0;
  // Iterate over the reduction dimension
  for (int k = 1; k < K; ++k) {
    scalar_t val = slice_ptr[k * inner_size];
    if (val < min_val) {
      min_val = val;
      min_idx = k;
    }
  }
  
  // Write result in output; the output is stored in row-major order over [outer, inner].
  output[static_cast<int64_t>(outer) * inner_size + inner] = min_idx;
}


at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
  int dims = x.dim();
  if (dim < 0) {
    dim += dims;
  }
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");
  
  // Compute outer_size = product of dimensions before the reduction dim.
  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  
  // K is the size of the dimension to reduce
  int K = static_cast<int>(x.size(dim));
  
  // Compute inner_size = product of dimensions after the reduction dim
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }
  
  // Prepare output shape: remove the reduction dimension from the original shape
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));
  
  // Launch the 2D kernel: gridDim.y = outer_size, gridDim.x covers inner dimension
  int threads_x = 256;
  dim3 block(threads_x);
  dim3 grid((inner_size + threads_x - 1) / threads_x, outer_size);
  
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    const scalar_t* x_data = x.data_ptr<scalar_t>();
    int64_t* output_data = output.data_ptr<int64_t>();
    argmin_2d_kernel<scalar_t><<<grid, block>>>(x_data, output_data, K, inner_size);
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
