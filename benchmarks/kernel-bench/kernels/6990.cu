#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <limits>

// This kernel optimizes global memory loads using __ldg for read-only access and aligns memory accesses.
template <typename scalar_t>
__global__ void argmin_optimized_memory_kernel(const scalar_t* __restrict__ x,
                                                int64_t* __restrict__ output,
                                                int K,
                                                int64_t inner_size) {
  int outer = blockIdx.y;
  int inner = blockIdx.x * blockDim.x + threadIdx.x;

  if(inner >= inner_size) return;

  const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

  scalar_t min_val = __ldg(&slice_start[0]);
  int min_index = 0;

  for (int k = 1; k < K; ++k) {
    scalar_t val = __ldg(&slice_start[k * inner_size]);
    if (val < min_val) {
      min_val = val;
      min_index = k;
    }
  }

  output[outer * inner_size + inner] = min_index;
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

  int dims = x.dim();
  if (dim < 0) {
    dim += dims;
  }
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  int K = static_cast<int>(x.size(dim));
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }

  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

  int threads = 256;
  dim3 block_dim(threads);
  dim3 grid_dim((inner_size + threads - 1) / threads, outer_size);

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    const scalar_t* x_data = x.data_ptr<scalar_t>();
    int64_t* output_data = output.data_ptr<int64_t>();
    argmin_optimized_memory_kernel<scalar_t><<<grid_dim, block_dim>>>(x_data, output_data, K, inner_size);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Argmin forward with optimized memory loads (CUDA)");
}
