#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Warp reduction functions
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_max(scalar_t val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Kernel that partitions each row across warps so that each warp accesses a contiguous segment
// This ensures that, within a warp, global memory accesses are coalesced
// The kernel performs a two-pass reduction: first to compute the maximum, then to compute the exp-sum
// Finally, it computes the log softmax output for the row

template <typename scalar_t>
__global__ void log_softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

  // Each block processes one row (batch element)
  const int row = blockIdx.x;
  const int warpSize = 32;
  const int warps_per_block = blockDim.x / warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  // Partition the row into contiguous chunks, one per warp
  int chunk = (dim_size + warps_per_block - 1) / warps_per_block; // ceiling division
  int start = warp_id * chunk;
  int end = start + chunk;
  if (end > dim_size) end = dim_size;

  const scalar_t* input_row = input + row * dim_size;

  // Compute the maximum value in this warp's chunk
  scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
  for (int i = start + lane; i < end; i += warpSize) {
    scalar_t val = __ldg(input_row + i);
    local_max = max(local_max, val);
  }
  local_max = warp_reduce_max(local_max);

  // Allocate shared memory for inter-warp reduction
  extern __shared__ char smem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
  // Each warp stores its local maximum at index 'warp_id'
  if (lane == 0)
    sdata[warp_id] = local_max;
  __syncthreads();

  // First warp reduces the warp-level maxima to compute the overall row maximum
  scalar_t row_max;
  if (threadIdx.x < warpSize) {
    if (threadIdx.x < warps_per_block)
      row_max = sdata[threadIdx.x];
    else
      row_max = -std::numeric_limits<scalar_t>::infinity();
    row_max = warp_reduce_max(row_max);
    if (threadIdx.x == 0)
      sdata[0] = row_max;
  }
  __syncthreads();
  row_max = sdata[0];

  // Second pass: compute the sum of exponentials in this warp's chunk
  scalar_t local_sum = 0;
  for (int i = start + lane; i < end; i += warpSize) {
    scalar_t val = __ldg(input_row + i);
    local_sum += exp(val - row_max);
  }
  local_sum = warp_reduce_sum(local_sum);

  // Store each warp's local sum in the second half of shared memory
  if (lane == 0)
    sdata[warp_id + warps_per_block] = local_sum;
  __syncthreads();

  // First warp reduces the warp-level sums to compute the overall row sum
  scalar_t row_sum;
  if (threadIdx.x < warpSize) {
    if (threadIdx.x < warps_per_block)
      row_sum = sdata[threadIdx.x + warps_per_block];
    else
      row_sum = 0;
    row_sum = warp_reduce_sum(row_sum);
    if (threadIdx.x == 0)
      sdata[0] = row_sum;
  }
  __syncthreads();
  row_sum = sdata[0];
  scalar_t log_row_sum = log(row_sum);

  // Final pass: write the log softmax output for the warp's chunk
  for (int i = start + lane; i < end; i += warpSize) {
    scalar_t val = __ldg(input_row + i);
    scalar_t result = (val - row_max) - log_row_sum;
    output[row * dim_size + i] = result;
  }
}

// Host function

torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 ||
              input.scalar_type() == torch::kFloat64,
              "input must be float32 or float64");

  int64_t ndim = input.dim();
  TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
  dim = dim >= 0 ? dim : dim + ndim;

  // Permute to bring the selected dimension to the last position
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

  // Choose the number of threads per block (a multiple of warp size)
  int threads = 128;
  if (threads > dim_size) {
    threads = ((dim_size + 31) / 32) * 32;
  }
  int blocks = batch_size;

  // Calculate shared memory size: we need space for 2 arrays of size (warps_per_block)
  int warps_per_block = threads / 32;
  size_t shmem_size = (warps_per_block * 2) * sizeof(float); // works for float and adjusted later for other types

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
    shmem_size = (warps_per_block * 2) * sizeof(scalar_t);
    log_softmax_forward_kernel<scalar_t><<<blocks, threads, shmem_size>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        dim_size);
  }));

  // Inverse permute to restore original layout
  std::vector<int64_t> inverse_permute_dims(ndim);
  for (size_t i = 0; i < permute_dims.size(); ++i) {
    inverse_permute_dims[permute_dims[i]] = i;
  }
  output = output.permute(inverse_permute_dims);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward (CUDA)");
}
