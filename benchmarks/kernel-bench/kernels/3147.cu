#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Define warp size
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Warp-level reduction for maximum
template <typename scalar_t>
__device__ inline scalar_t warpReduceMax(scalar_t val) {
  // Fully use 32 threads in a warp
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// Warp-level reduction for sum
template <typename scalar_t>
__device__ inline scalar_t warpReduceSum(scalar_t val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Optimized LogSoftmax kernel with reduced __syncthreads(), using warp shuffle reductions
// BLOCK_SIZE is the number of threads per block (e.g., 512)
template <typename scalar_t, int BLOCK_SIZE>
__global__ void log_softmax_forward_kernel_opt(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

  // Each block handles one batch element (one row)
  int batch_idx = blockIdx.x;
  const scalar_t* in_row = input + batch_idx * dim_size;
  scalar_t* out_row = output + batch_idx * dim_size;
  int tid = threadIdx.x;

  // Phase 1: Compute maximum value in the row using grid-stride loop
  scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
  for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
    local_max = max(local_max, in_row[i]);
  }
  // Intra-warp reduction
  local_max = warpReduceMax(local_max);

  // Each warp writes its partial max to shared memory
  __shared__ scalar_t sdata_max[BLOCK_SIZE / WARP_SIZE];
  int warpId = tid / WARP_SIZE;
  if ((tid % WARP_SIZE) == 0) {
    sdata_max[warpId] = local_max;
  }
  // Synchronize only to ensure partial results are written
  __syncthreads();

  // Let the first warp reduce the warp-level partial maxima
  if (tid < (BLOCK_SIZE / WARP_SIZE)) {
    local_max = sdata_max[tid];
  }
  if (tid < (BLOCK_SIZE / WARP_SIZE)) {
    local_max = warpReduceMax(local_max);
    if (tid == 0) {
      sdata_max[0] = local_max;
    }
  }
  // Synchronize so that all threads can read the final max
  __syncthreads();
  scalar_t final_max = sdata_max[0];

  // Phase 2: Compute sum of exp(x - final_max) in a similar fashion
  scalar_t local_sum = 0;
  for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
    local_sum += exp(in_row[i] - final_max);
  }
  local_sum = warpReduceSum(local_sum);

  __shared__ scalar_t sdata_sum[BLOCK_SIZE / WARP_SIZE];
  if ((tid % WARP_SIZE) == 0) {
    sdata_sum[warpId] = local_sum;
  }
  __syncthreads();
  if (tid < (BLOCK_SIZE / WARP_SIZE)) {
    local_sum = sdata_sum[tid];
  }
  if (tid < (BLOCK_SIZE / WARP_SIZE)) {
    local_sum = warpReduceSum(local_sum);
    if (tid == 0) {
      sdata_sum[0] = local_sum;
    }
  }
  __syncthreads();
  scalar_t total_sum = sdata_sum[0];
  scalar_t log_sum = log(total_sum);

  // Phase 3: Compute LogSoftmax output
  for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
    out_row[i] = (in_row[i] - final_max) - log_sum;
  }
}

// Host function that permutes tensor dimensions, launches the kernel, and inverse permutes the result
torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
      "input must be float32 or float64");

  int64_t ndim = input.dim();
  TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
  dim = dim >= 0 ? dim : dim + ndim;

  // Permute input to bring target dimension to the last axis
  std::vector<int64_t> permute_dims;
  for (int64_t i = 0; i < ndim; ++i) {
    if (i != dim) {
      permute_dims.push_back(i);
    }
  }
  permute_dims.push_back(dim);
  input = input.permute(permute_dims).contiguous();

  int64_t batch_size = input.numel() / input.size(-1);
  int64_t dim_size = input.size(-1);

  auto output = torch::empty_like(input);

  // Set block size to 512 threads
  const int BLOCK_SIZE = 512;
  int blocks = batch_size;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
    log_softmax_forward_kernel_opt<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        dim_size);
  }));

  // Inverse permutation to restore original layout
  std::vector<int64_t> inverse_permute_dims(ndim);
  for (size_t i = 0; i < permute_dims.size(); ++i) {
    inverse_permute_dims[permute_dims[i]] = i;
  }
  output = output.permute(inverse_permute_dims);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &log_softmax_cuda_forward, "Optimized LogSoftmax forward (CUDA) with reduced synchronizations");
}
