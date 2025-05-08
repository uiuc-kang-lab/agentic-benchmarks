#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// Kernel that uses a stride loop to handle workloads larger than available threads and unrolls the reduction loop over K.

template <typename scalar_t>
__global__ void argmin_strided_unroll_kernel(const scalar_t* __restrict__ x,
                                               int64_t* __restrict__ output,
                                               int K,
                                               int64_t inner_size) {
  // Each block in the y-dimension corresponds to one outer slice.
  int outer = blockIdx.y;
  // Compute initial thread index for the inner dimension.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Stride loop: Each thread may process multiple inner indices.
  for (int inner = idx; inner < inner_size; inner += gridDim.x * blockDim.x) {
    // Compute starting pointer for the current slice [outer, :, inner].
    const scalar_t* slice = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;
    
    // Initialize minimum value and index.
    scalar_t min_val = slice[0];
    int min_idx = 0;

    // Unroll the reduction loop over K by a factor of 4.
    int k = 1;
    // Determine the unrolled loop bound. For K>=1, we unroll groups of 4 starting from index 1.
    int loop_bound = 1 + ((K - 1) / 4) * 4;
    for (; k < loop_bound; k += 4) {
      scalar_t v0 = slice[k * inner_size];
      scalar_t v1 = slice[(k + 1) * inner_size];
      scalar_t v2 = slice[(k + 2) * inner_size];
      scalar_t v3 = slice[(k + 3) * inner_size];
      if (v0 < min_val) { min_val = v0; min_idx = k; }
      if (v1 < min_val) { min_val = v1; min_idx = k + 1; }
      if (v2 < min_val) { min_val = v2; min_idx = k + 2; }
      if (v3 < min_val) { min_val = v3; min_idx = k + 3; }
    }
    // Process any remaining elements not covered by the unrolled loop.
    for (; k < K; ++k) {
      scalar_t v = slice[k * inner_size];
      if (v < min_val) {
        min_val = v;
        min_idx = k;
      }
    }
    
    // Store the index of the minimum value.
    output[outer * inner_size + inner] = min_idx;
  }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  // Validate that the input tensor is on CUDA.
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

  int dims = x.dim();
  if (dim < 0) {
    dim += dims;
  }
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  // Compute outer_size as the product of dimensions before the reduction dimension.
  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  
  // K is the size of the reduction dimension.
  int K = static_cast<int>(x.size(dim));
  
  // inner_size is the product of dimensions after the reduction dimension.
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }

  // Build the output shape by removing the reduction dimension.
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

  // Configure a 2D grid: grid.y corresponds to outer slices, grid.x to inner indices.
  int threads = 256;
  dim3 block_dim(threads);
  dim3 grid_dim((inner_size + threads - 1) / threads, outer_size);

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    const scalar_t* x_data = x.data_ptr<scalar_t>();
    int64_t* output_data = output.data_ptr<int64_t>();
    argmin_strided_unroll_kernel<scalar_t><<<grid_dim, block_dim>>>(x_data, output_data, K, inner_size);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
  }
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Argmin forward with strided loop and loop unrolling (CUDA)");
}
