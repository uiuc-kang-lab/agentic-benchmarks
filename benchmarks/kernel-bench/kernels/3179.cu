#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Kernel that leverages shared memory to store each input row,
// reducing global memory accesses for repeated reads during reductions.
// The shared memory is partitioned into two arrays: one for the input row copy
// and one for reduction (max and sum) values. This minimizes global memory latency
// and avoids race conditions during reduction.


template <typename scalar_t>
__global__ void shared_mem_logsoftmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim_size) {

  // Dynamically allocated shared memory partitioned as follows:
  // s_row: holds the entire input row (dim_size elements)
  // s_reduce: holds blockDim.x elements for reduction
  extern __shared__ char smem[];
  scalar_t* s_row = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_reduce = s_row + dim_size;

  int batch_idx = blockIdx.x;
  const scalar_t* row_in = input + batch_idx * dim_size;
  scalar_t* row_out = output + batch_idx * dim_size;

  // Phase 1: Copy the entire input row from global memory to shared memory
  for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
    s_row[i] = row_in[i];
  }
  __syncthreads();

  // Phase 2: Compute the maximum value in the row using shared memory reduction
  scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
  for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
    local_max = (s_row[i] > local_max) ? s_row[i] : local_max;
  }
  s_reduce[threadIdx.x] = local_max;
  __syncthreads();

  // Reduce to get the maximum value across the block
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] = (s_reduce[threadIdx.x] > s_reduce[threadIdx.x + stride]) ? s_reduce[threadIdx.x] : s_reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  scalar_t max_val = s_reduce[0];
  __syncthreads();

  // Phase 3: Compute the sum of exp(x - max_val) using the shared memory copy
  scalar_t local_sum = 0;
  for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
    local_sum += exp(s_row[i] - max_val);
  }
  s_reduce[threadIdx.x] = local_sum;
  __syncthreads();

  // Reduce to get the total sum
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  scalar_t total_sum = s_reduce[0];
  scalar_t log_sum = log(total_sum);
  __syncthreads();

  // Phase 4: Write the final LogSoftmax output back to global memory
  for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
    row_out[i] = (s_row[i] - max_val) - log_sum;
  }
}


// Host function: Permute the input so that the softmax dimension is last, launch the kernel,
// and then inversely permute the output to restore the original layout.

torch::Tensor shared_mem_logsoftmax_cuda_forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
              "input must be float32 or float64");

  int64_t ndim = input.dim();
  TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");
  dim = dim >= 0 ? dim : dim + ndim;

  // Permute input so that target dimension is the last dimension
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

  // Choose a block size. For shared memory copy, a fixed block size (e.g., 256) is often effective.
  int block_size = 256;
  if (dim_size < block_size) {
    block_size = dim_size;
  }
  const int blocks = batch_size;

  // Calculate required shared memory size:
  // We need space for the input row (dim_size elements) plus one reduction array (block_size elements).
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "shared_mem_logsoftmax_cuda_forward", ([&] {
    size_t shared_mem_bytes = (dim_size + block_size) * sizeof(scalar_t);
    shared_mem_logsoftmax_kernel<scalar_t><<<blocks, block_size, shared_mem_bytes>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        dim_size);
  }));

  // Inverse permute to restore original tensor layout
  std::vector<int64_t> inverse_permute_dims(ndim);
  for (size_t i = 0; i < permute_dims.size(); ++i) {
    inverse_permute_dims[permute_dims[i]] = i;
  }
  output = output.permute(inverse_permute_dims);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &shared_mem_logsoftmax_cuda_forward, "Shared Memory LogSoftmax forward (CUDA)");
}
