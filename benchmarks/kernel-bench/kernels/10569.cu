#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel minimizes warp divergence by eliminating any conditional branching within loops.
// Instead of looping over non-fixed sizes, it assumes a fixed maximum segment per thread dynamically determined during launch configuration.

// Helper macro to reduce branching by looping over fixed size in parallel scan
#define MAX_SEGMENT_SIZE 512

// This kernel processes one cumulative product chain per block using two-pass scanning.
// It reduces warp divergence by refactoring conditional logic for uniform control flow.
template <typename scalar_t>
__global__ void cumprod_minimized_divergence_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

  int chain_id = blockIdx.x;  // one block per chain
  if (chain_id >= total_chains) return;

  int batch_idx = chain_id / stride;
  int in_idx = chain_id % stride;
  int64_t base = batch_idx * (dim_size * stride) + in_idx;

  int t = threadIdx.x;

  // Shared memory to store each thread's local product
  __shared__ scalar_t s_local[MAX_SEGMENT_SIZE];
  scalar_t* s_offset = s_local + blockDim.x;  // Offset for exclusive scan

  scalar_t local_prod = 1;
  int chunk = (dim_size + blockDim.x - 1) / blockDim.x;
  int start = t * chunk;
  int end = min(start + chunk, dim_size);

  // Compute local segment product
  for (int i = start; i < end; ++i) {
    int idx = (base + i * stride);
    local_prod *= input[idx];
  }
  s_local[t] = local_prod;
  __syncthreads();

  // Exclusive scan to compute offset
  scalar_t offset = 1;
  if (t == 0) {
    offset = 1;
    for (int i = 0; i < blockDim.x; ++i) {
      scalar_t last_offset = offset;
      offset *= s_local[i];
      s_offset[i] = last_offset;
    }
  }
  __syncthreads();

  offset = s_offset[t];
  scalar_t prod = offset;
  // Process each segment with the offset
  for (int i = start; i < end; ++i) {
    int64_t idx = base + i * stride;
    prod *= input[idx];
    output[idx] = prod;
  }
}


torch::Tensor cumprod_cuda_divergence_minimized_forward(torch::Tensor input, int64_t dim) {
  auto output = torch::empty_like(input);
  
  auto sizes = input.sizes();
  auto strides = input.strides();
  int64_t dim_size = sizes[dim];
  int64_t stride_val = strides[dim];
  int64_t total_chains = input.numel() / dim_size;

  int threads = MAX_SEGMENT_SIZE;
  dim3 blocks(total_chains);
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_minimized_divergence", ([&] {
    cumprod_minimized_divergence_kernel<scalar_t><<<blocks, threads>>>(
      output.data_ptr<scalar_t>(),
      input.data_ptr<scalar_t>(),
      dim_size,
      stride_val,
      total_chains
    );
  }));
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cumprod_cuda_divergence_minimized_forward, "Minimized divergence cumulative product forward with two-pass scan (CUDA)");
}
