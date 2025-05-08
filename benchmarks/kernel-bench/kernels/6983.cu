#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <limits>

// New CUDA kernel that performs a parallel reduction over the K dimension using shared memory.
// The tensor is conceptually reshaped as [outer, K, inner] where:
//   outer = product_{i=0}^{dim-1} x.size(i)
//   K     = x.size(dim)
//   inner = product_{i=dim+1}^{D-1} x.size(i)
// Each block is assigned one (outer, inner) pair. Threads within the block cooperatively reduce over the K dimension.

template <typename scalar_t>
__global__ void argmin_parallel_kernel(const scalar_t* __restrict__ x,
                                         int64_t* __restrict__ output,
                                         int K,
                                         int64_t inner_size) {
  // Map grid dimensions to the output spatial dimensions:
  // gridDim.y indexes the outer dimension and gridDim.x indexes the inner dimension.
  int outer = blockIdx.y;
  int inner = blockIdx.x;

  // Ensure inner index is within bounds (outer is always in range since gridDim.y is set accordingly)
  if (inner >= inner_size)
    return;

  // Compute the starting pointer for the current slice
  // The data layout is interpreted as: [outer, K, inner]
  const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

  // Each block uses blockDim.x threads to cooperatively reduce the K dimension.
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  // Initialize each thread's local candidate with the maximum possible value
  scalar_t local_min = std::numeric_limits<scalar_t>::max();
  int local_idx = 0;

  // Each thread processes multiple elements in the K dimension in a strided manner.
  for (int k = tid; k < K; k += block_size) {
    // Because of the layout, the element corresponding to k is at offset k * inner_size
    scalar_t val = slice_start[k * inner_size];
    if (val < local_min) {
      local_min = val;
      local_idx = k;
    }
  }

  // Allocate dynamic shared memory for reduction.
  // We allocate an array of local minimum values and an array of corresponding indices.
  extern __shared__ char shared_mem[];
  scalar_t* sdata_val = reinterpret_cast<scalar_t*>(shared_mem);
  int* sdata_idx = reinterpret_cast<int*>(shared_mem + block_size * sizeof(scalar_t));

  sdata_val[tid] = local_min;
  sdata_idx[tid] = local_idx;
  __syncthreads();

  // Perform parallel reduction in shared memory to get the minimum value and its index
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (sdata_val[tid + stride] < sdata_val[tid]) {
        sdata_val[tid] = sdata_val[tid + stride];
        sdata_idx[tid] = sdata_idx[tid + stride];
      }
    }
    __syncthreads();
  }

  // The first thread now holds the index of the minimum element along the K dimension
  if (tid == 0) {
    output[outer * inner_size + inner] = sdata_idx[0];
  }
}


at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  // Ensure the input tensor is a CUDA tensor
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

  int dims = x.dim();
  if (dim < 0) {
    dim += dims;
  }
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  // Compute the dimensions for the conceptual reshaping: [outer, K, inner]
  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  int K = static_cast<int>(x.size(dim));
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }

  // The output tensor has the same shape as the input tensor except that the reduction dimension is removed
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

  // Choose the number of threads for the reduction along the K dimension.
  // If K is small, use K threads; otherwise cap at 256 threads per block.
  int threads = (K < 256 ? K : 256);

  // Configure the grid such that each block processes a single output element (one (outer, inner) pair).
  // gridDim.x covers the inner dimension and gridDim.y covers the outer dimension.
  dim3 block_dim(threads);
  dim3 grid_dim(inner_size, outer_size);

  // Calculate dynamic shared memory: for each thread, space for a scalar_t and an int
  size_t shared_memory_size = threads * (sizeof(scalar_t) + sizeof(int));

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    const scalar_t* x_data = x.data_ptr<scalar_t>();
    int64_t* output_data = output.data_ptr<int64_t>();
    argmin_parallel_kernel<scalar_t><<<grid_dim, block_dim, shared_memory_size>>>(x_data, output_data, K, inner_size);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Argmin forward with parallel reduction (CUDA)");
}
