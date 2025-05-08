#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <limits>

// Kernel that uses grid-stride loops with warp-level intrinsics (__shfl_down_sync) to evenly distribute workload across threads and blocks.
// Each block processes multiple slices (of the reduction dimension) if necessary. 
// Intra-warp reduction is used to reduce overhead and ensure that each thread contributes evenly.

template <typename scalar_t>
__global__ void argmin_warp_shuffle_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {
  // Total number of slices to reduce
  int64_t total_slices = outer_size * inner_size;
  
  // Use grid-stride loop to let each block process multiple slices
  for (int slice = blockIdx.x; slice < total_slices; slice += gridDim.x) {
    // Compute the outer and inner indices for the current slice
    int64_t outer = slice / inner_size;
    int64_t inner = slice % inner_size;
    
    // Initialize local minimum and corresponding index
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    int local_idx = 0;

    // Each thread processes a subset of the reduction dimension elements
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
      // Compute index: tensor is viewed as [outer_size, K, inner_size]
      int64_t index = outer * (static_cast<int64_t>(K) * inner_size) + k * inner_size + inner;
      scalar_t val = __ldg(&x[index]);
      if (val < local_min) {
        local_min = val;
        local_idx = k;
      }
    }

    // Intra-warp reduction using warp shuffle
    unsigned int full_mask = 0xFFFFFFFF;
    int lane = threadIdx.x & 31; // thread index within warp
    
    // Reduce within each warp
    for (int offset = 16; offset > 0; offset /= 2) {
      scalar_t other = __shfl_down_sync(full_mask, local_min, offset);
      int other_idx = __shfl_down_sync(full_mask, local_idx, offset);
      if (other < local_min) {
        local_min = other;
        local_idx = other_idx;
      }
    }

    // Each warp's lane 0 writes its result to shared memory
    __shared__ scalar_t s_warp_min[32];
    __shared__ int s_warp_idx[32];
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
      s_warp_min[warp_id] = local_min;
      s_warp_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // Let the first warp reduce the results from all warps in the block
    int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
      // Only use the first 'num_warps' lanes; others get a neutral value
      scalar_t final_min = (lane < num_warps) ? s_warp_min[lane] : std::numeric_limits<scalar_t>::max();
      int final_idx = (lane < num_warps) ? s_warp_idx[lane] : 0;

      for (int offset = 16; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(full_mask, final_min, offset);
        int other_idx = __shfl_down_sync(full_mask, final_idx, offset);
        if (other < final_min) {
          final_min = other;
          final_idx = other_idx;
        }
      }
      if (lane == 0) {
        output[slice] = final_idx;
      }
    }
  }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
  
  int dims = x.dim();
  if (dim < 0) dim += dims;
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");
  
  // Compute sizes: view tensor as shape [outer_size, K, inner_size]
  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  int K = static_cast<int>(x.size(dim));
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }
  
  // Output tensor has reduced dimension removed
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));
  
  // Total number of slices
  int64_t total_slices = outer_size * inner_size;
  
  // Launch configuration: use a fixed number of blocks to evenly distribute workload with grid-stride looping
  int blocks = 1024; 
  int threads = 256;

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    const scalar_t* x_data = x.data_ptr<scalar_t>();
    int64_t* output_data = output.data_ptr<int64_t>();
    argmin_warp_shuffle_kernel<scalar_t><<<blocks, threads>>>(
        x_data, output_data, K, outer_size, inner_size);
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
