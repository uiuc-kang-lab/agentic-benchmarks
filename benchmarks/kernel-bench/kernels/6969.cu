#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <limits>

// This kernel uses a two-level reduction: first, each thread processes a strided subset of the K elements
// and then performs a warp-level reduction using shuffle intrinsics (which requires no synchronization).
// Only one __syncthreads() call is used when writing warp-level results to shared memory for the final block-level reduction.
// Each CUDA block processes one complete slice corresponding to an output element (from [outer, K, inner]).


template <typename scalar_t>
__global__ void argmin_shared_reduce_kernel(const scalar_t* __restrict__ x,
                                              int64_t* __restrict__ output,
                                              int K,
                                              int64_t inner_size) {
  // Each block handles one output slice: mapping thread block index to (outer, inner) indices.
  int slice_id = blockIdx.x;  // slice_id in [0, outer_size * inner_size)
  int outer = slice_id / inner_size;
  int inner = slice_id % inner_size;

  // Pointer to the start of the current slice in the K dimension.
  const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

  int tid = threadIdx.x;
  scalar_t local_min;
  int local_idx;

  // Initialize each thread's local minimum if it has at least one element to process.
  if (tid < K) {
    local_min = slice_start[tid * inner_size];
    local_idx = tid;
    // Process remaining elements in a strided manner
    for (int k = tid + blockDim.x; k < K; k += blockDim.x) {
      scalar_t val = slice_start[k * inner_size];
      if (val < local_min) {
        local_min = val;
        local_idx = k;
      }
    }
  } else {
    // Threads that don't have any element to process get a neutral value
    local_min = std::numeric_limits<scalar_t>::max();
    local_idx = K;  // An invalid index
  }

  // Use warp-level reduction with shuffle intrinsics; no __syncthreads needed within a warp.
  unsigned int mask = 0xffffffff;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    scalar_t other_min = __shfl_down_sync(mask, local_min, offset);
    int other_idx = __shfl_down_sync(mask, local_idx, offset);
    if (other_min < local_min) {
      local_min = other_min;
      local_idx = other_idx;
    }
  }

  // Each warp writes its result to shared memory.
  __shared__ scalar_t shared_min[32];   // Enough to hold results from up to 32 warps.
  __shared__ int shared_idx[32];
  int lane = tid & (warpSize - 1);
  int warpId = tid / warpSize;
  if (lane == 0) {
    shared_min[warpId] = local_min;
    shared_idx[warpId] = local_idx;
  }

  // Synchronize only once to ensure all warp leaders have written their values.
  __syncthreads();

  // Final reduction: let thread 0 process the warp-level results.
  if (tid == 0) {
    scalar_t final_min = shared_min[0];
    int final_idx = shared_idx[0];
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    for (int i = 1; i < numWarps; i++) {
      if (shared_min[i] < final_min) {
        final_min = shared_min[i];
        final_idx = shared_idx[i];
      }
    }
    output[slice_id] = final_idx;
  }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  // Ensure the input tensor is on CUDA
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

  int dims = x.dim();
  if (dim < 0) {
    dim += dims;
  }
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  // Reshape the tensor to a conceptual shape: [outer, K, inner]
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

  // Each block will process one slice (an output element), so the number of blocks equals outer_size * inner_size.
  int64_t total_slices = outer_size * inner_size;
  int threads = 256;
  dim3 block_dim(threads);
  dim3 grid_dim(total_slices);

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    const scalar_t* x_data = x.data_ptr<scalar_t>();
    int64_t* output_data = output.data_ptr<int64_t>();
    argmin_shared_reduce_kernel<scalar_t><<<grid_dim, block_dim>>>(x_data, output_data, K, inner_size);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Argmin forward with shared memory reduction (CUDA)");
}
