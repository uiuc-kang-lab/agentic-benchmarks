#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Kernel that uses warp-level primitives and shared memory atomics (only within block) to reduce
// the number of synchronizations and avoid global atomic contention. Each block processes one row.

template <typename scalar_t>
__global__ void warp_atomic_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {
  // Each block handles one row (batch element)
  int row = blockIdx.x;
  const scalar_t* input_row = input + row * dim_size;

  // Phase 1: Compute maximum value in the row using warp-level reduction
  scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();

  // Each thread processes a chunk of the row
  for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
    local_max = max(local_max, input_row[idx]);
  }

  // Reduce within each warp using shuffle
  unsigned int mask = 0xffffffff;
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(mask, local_max, offset);
    local_max = max(local_max, other);
  }

  // Each warp's leader writes its result to shared memory
  __shared__ scalar_t warp_max[32]; // support up to 32 warps per block
  int warp_id = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;
  if (lane == 0) {
    warp_max[warp_id] = local_max;
  }
  __syncthreads();

  // Thread 0 computes the block-wide maximum from warp_max values
  scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
  if (threadIdx.x == 0) {
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    for (int i = 0; i < num_warps; i++) {
      max_val = max(max_val, warp_max[i]);
    }
    warp_max[0] = max_val;  // store global max for broadcast
  }
  __syncthreads();
  max_val = warp_max[0];

  // Phase 2: Compute sum of exp(x - max_val) using atomics in shared memory to combine warp results
  scalar_t local_sum = 0;
  for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
    local_sum += exp(input_row[idx] - max_val);
  }
  // Warp-level reduction for local_sum
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(mask, local_sum, offset);
  }

  // Use a shared memory variable to accumulate warp sums with atomicAdd (minimizing global atomics)
  __shared__ scalar_t sum_shared;
  if (threadIdx.x == 0) {
    sum_shared = 0;
  }
  __syncthreads();
  if (lane == 0) {
    atomicAdd(&sum_shared, local_sum);
  }
  __syncthreads();
  scalar_t total_sum = sum_shared;
  scalar_t log_sum = log(total_sum);

  // Phase 3: Final computation of LogSoftmax: (x - max_val) - log(sum(exp(x - max_val)))
  for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
    output[row * dim_size + idx] = (input_row[idx] - max_val) - log_sum;
  }
}


// Host function: Permutes input tensor so the specified dimension is last, launches kernel, then inversely permutes output

torch::Tensor warp_atomic_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
      "input must be float32 or float64");

  int64_t ndim = input.dim();
  TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
  dim = dim >= 0 ? dim : dim + ndim;

  // Permute input so that the reduction dimension becomes the last dimension
  std::vector<int64_t> permute_dims;
  for (int64_t i = 0; i < ndim; ++i) {
    if (i != dim) {
      permute_dims.push_back(i);
    }
  }
  permute_dims.push_back(dim);
  input = input.permute(permute_dims).contiguous();

  int64_t num_rows = input.numel() / input.size(-1);
  int64_t dim_size = input.size(-1);

  auto output = torch::empty_like(input);

  // Launch one block per row with a fixed number of threads (e.g., 256)
  int threads = 256;
  dim3 blocks(num_rows);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "warp_atomic_logsoftmax_cuda_forward", ([&] {
    warp_atomic_logsoftmax_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        dim_size);
  }));

  // Inverse permute to restore original tensor shape
  std::vector<int64_t> inverse_permute_dims(ndim);
  for (size_t i = 0; i < permute_dims.size(); ++i) {
    inverse_permute_dims[permute_dims[i]] = i;
  }
  output = output.permute(inverse_permute_dims);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &warp_atomic_logsoftmax_cuda_forward, "Warp Atomic LogSoftmax forward (CUDA)");
}
