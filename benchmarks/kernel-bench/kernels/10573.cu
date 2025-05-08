#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Constant memory for storing read-only input data that fits
// Limited to 64KB in size, so should be used judiciously.
__constant__ float const_input[65536]; // 256 * 256 max dim_size * stride

// This kernel processes one cumulative product chain per block.
// We aim to store frequently accessed read-only data in constant memory for faster access.
// For this, we need to ensure the data fits into the allowed constant memory size.

template <typename scalar_t>
__global__ void cumprod_constant_memory_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

  int chain_id = blockIdx.x;  // one block per chain
  if (chain_id >= total_chains) return;

  // Decode the chain to get batch index and in-dimension index
  int batch_idx = chain_id / stride;
  int in_idx = chain_id % stride;
  int64_t base = batch_idx * (dim_size * stride) + in_idx;

  int t = threadIdx.x;
  int T = blockDim.x;
  // Divide the chain evenly among threads
  int chunk = (dim_size + T - 1) / T;

  // Shared memory to store each thread's local product
  __shared__ scalar_t s_local[1024];  // assuming blockDim.x <= 1024

  int start = t * chunk;
  int end = start + chunk;
  if (end > dim_size) end = dim_size;

  // Load the input data into constant memory if necessary
  if (t == 0) {
    for (int i = start; i < end; i++) {
      int64_t idx = base + i * stride;
      const_input[i] = input[idx];
    }
  }
  __syncthreads();

  // First pass: each thread computes the product of its segment sequentially
  scalar_t local_prod = 1;
  for (int i = start; i < end; i++) {
    local_prod = local_prod * const_input[i];
  }
  s_local[t] = local_prod;
  __syncthreads();

  // Compute the exclusive product (offset) for this thread: product of all segments from threads with id < t
  scalar_t offset = 1;
  for (int i = 0; i < t; i++) {
    offset = offset * s_local[i];
  }
  __syncthreads();

  // Second pass: re-read the input and compute the cumulative product for the assigned segment,
  // starting with the offset computed from previous threads
  scalar_t prod = offset;
  for (int i = start; i < end; i++) {
    prod = prod * const_input[i];
    int64_t idx = base + i * stride;
    output[idx] = prod;
  }
}


// CUDA forward function that launches one block per cumulative product chain

torch::Tensor cumprod_cuda_constant_forward(torch::Tensor input, int64_t dim) {
  auto output = torch::empty_like(input);
  
  // Extract tensor info
  auto sizes = input.sizes();
  auto strides = input.strides();
  int64_t dim_size = sizes[dim];
  int64_t stride_val = strides[dim];
  // total number of independent cumulative product chains
  int64_t total_chains = input.numel() / dim_size;
  
  // Launch one block per chain. We choose 256 threads per block to distribute the workload evenly along the chain.
  int threads = 256;
  dim3 blocks(total_chains);
  dim3 threads_per_block(threads);
  
  // Pre-load constant memory with input
  if ((dim_size * stride_val) < 65536) { // check if fits within constant memory size
      cudaMemcpyToSymbol(const_input, input.data_ptr<scalar_t>(), dim_size * stride_val * sizeof(scalar_t));
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_constant", ([&] {
    cumprod_constant_memory_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
  m.def("forward", &cumprod_cuda_constant_forward, "Cumulative product forward with constant memory (CUDA)");
}
