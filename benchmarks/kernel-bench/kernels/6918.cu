#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// Declare constant memory for frequently accessed parameters
__constant__ int cK;
__constant__ int64_t c_inner_size;

// CUDA kernel using constant memory for K and inner_size
template <typename scalar_t>
__global__ void argmin_kernel_const(const scalar_t* __restrict__ x,
                                      int64_t* __restrict__ output,
                                      int64_t outer_size) {
  // Compute a global thread index (using 64-bit arithmetic)
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= outer_size * c_inner_size) return;
  
  // Decompose idx into outer and inner indices
  int64_t outer = idx / c_inner_size;
  int64_t inner = idx % c_inner_size;
  
  // Access the corresponding slice. Data layout is: [outer_size, K, inner_size]
  const scalar_t* slice_start = x + outer * (static_cast<int64_t>(cK) * c_inner_size) + inner;

  // Initialize with the first element in the K-dimension
  scalar_t min_val = slice_start[0];
  int min_index = 0;
  
  // Iterate over the K dimension
  for (int k = 1; k < cK; ++k) {
    scalar_t val = slice_start[k * c_inner_size];
    if (val < min_val) {
      min_val = val;
      min_index = k;
    }
  }
  
  output[idx] = min_index;
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  // Ensure input is a CUDA tensor
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

  int dims = x.dim();
  if (dim < 0) {
    dim += dims;
  }
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  // Compute outer_size, K, and inner_size based on the reduction dimension
  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  int K = static_cast<int>(x.size(dim));
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }

  // Prepare output tensor (shape of input with reduction dimension removed)
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

  // Copy constant parameters to constant memory
  cudaError_t err;
  err = cudaMemcpyToSymbol(cK, &K, sizeof(int));
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to copy K to constant memory: ") + cudaGetErrorString(err));
  }
  err = cudaMemcpyToSymbol(c_inner_size, &inner_size, sizeof(int64_t));
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to copy inner_size to constant memory: ") + cudaGetErrorString(err));
  }

  // Launch kernel
  int64_t total_slices = outer_size; // Each slice corresponds to [K, inner_size]
  // Total elements to process are outer_size * inner_size
  int64_t total_elements = outer_size * inner_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    const scalar_t* x_data = x.data_ptr<scalar_t>();
    int64_t* output_data = output.data_ptr<int64_t>();
    argmin_kernel_const<scalar_t><<<blocks, threads>>>(x_data, output_data, outer_size);
  }));

  // Check for any kernel launch errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Argmin forward with constant memory (CUDA)");
}
