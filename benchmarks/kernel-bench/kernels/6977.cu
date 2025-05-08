#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <limits>

// This kernel uses a parallel reduction with warp shuffle intrinsics to compute the argmin along the K dimension.
// Each block is assigned one output element (one (outer, inner) pair), and the K elements are reduced cooperatively by the threads.
// The reduction first uses warp-level shuffles (which do not require __syncthreads) and then one __syncthreads() to combine results from different warps.

template <typename scalar_t>
__global__ void argmin_parallel_reduction_kernel(const scalar_t* __restrict__ x,
                                                   int64_t* __restrict__ output,
                                                   int K,
                                                   int64_t inner_size) {
  // Determine the output element that this block is responsible for.
  int outer = blockIdx.y;
  int inner = blockIdx.x;
  // Base pointer for the slice corresponding to this (outer, inner) pair.
  const scalar_t* slice = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

  int tid = threadIdx.x;
  scalar_t local_min;
  int local_index;

  // Each thread computes a partial minimum over elements in the K dimension, using a stride of blockDim.x.
  if (tid < K) {
    local_min = slice[tid * inner_size];
    local_index = tid;
    for (int t = tid + blockDim.x; t < K; t += blockDim.x) {
      scalar_t val = slice[t * inner_size];
      if (val < local_min) {
        local_min = val;
        local_index = t;
      }
    }
  } else {
    local_min = std::numeric_limits<scalar_t>::max();
    local_index = 0;
  }

  // Intra-warp reduction using shuffle intrinsics (no __syncthreads needed here).
  unsigned int mask = 0xffffffff;
  int lane = tid & 31;
  int warpId = tid >> 5;  // divide by 32
  for (int offset = 16; offset > 0; offset /= 2) {
    scalar_t other_min = __shfl_down_sync(mask, local_min, offset);
    int other_index = __shfl_down_sync(mask, local_index, offset);
    if (other_min < local_min) {
      local_min = other_min;
      local_index = other_index;
    }
  }

  // Allocate dynamic shared memory to store each warp's results.
  extern __shared__ char shared[];
  scalar_t* smin = reinterpret_cast<scalar_t*>(shared);
  int* smin_idx = reinterpret_cast<int*>(shared + ((blockDim.x + 31) / 32) * sizeof(scalar_t));

  // Warp leaders store the result of their warp reduction to shared memory.
  if (lane == 0) {
    smin[warpId] = local_min;
    smin_idx[warpId] = local_index;
  }
  __syncthreads();

  // Final reduction from the warp-level results is performed by the first warp.
  if (tid < (blockDim.x + 31) / 32) {
    local_min = smin[lane];
    local_index = smin_idx[lane];
    for (int offset = 16; offset > 0; offset /= 2) {
      scalar_t other_min = __shfl_down_sync(mask, local_min, offset);
      int other_index = __shfl_down_sync(mask, local_index, offset);
      if (other_min < local_min) {
        local_min = other_min;
        local_index = other_index;
      }
    }
    if (lane == 0) {
      output[outer * inner_size + inner] = local_index;
    }
  }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
  
  int dims = x.dim();
  if (dim < 0) {
    dim += dims;
  }
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  // Reshape the tensor conceptually as [outer, K, inner]
  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  int K = static_cast<int>(x.size(dim));
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }

  // The output tensor has the reduction dimension removed
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

  // Grid configuration: each block computes one output element identified by (outer, inner).
  // gridDim.x = inner_size, gridDim.y = outer_size.
  int threads = 256;
  dim3 block_dim(threads);
  dim3 grid_dim(inner_size, outer_size);

  // Calculate dynamic shared memory size: one scalar_t and one int per warp.
  int warpsPerBlock = (threads + 31) / 32;
  
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    using scalar_t_internal = scalar_t;
    size_t shmem = warpsPerBlock * (sizeof(scalar_t_internal) + sizeof(int));
    const scalar_t_internal* x_data = x.data_ptr<scalar_t_internal>();
    int64_t* output_data = output.data_ptr<int64_t>();
    argmin_parallel_reduction_kernel<scalar_t_internal><<<grid_dim, block_dim, shmem>>>(
       x_data, output_data, K, inner_size);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Argmin forward with parallel reduction using minimal __syncthreads (CUDA)");
}
