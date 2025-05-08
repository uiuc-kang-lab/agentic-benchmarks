#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Combined kernel that performs cumulative product using a two-pass approach with per-thread loop unrolling
// and asynchronous stream memory transfers. Each block handles one cumulative product chain.

template <typename scalar_t>
__global__ void cumprod_combo_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

  // Each block processes one cumulative product chain
  int chain_id = blockIdx.x;
  if (chain_id >= total_chains) return;

  // Decode the chain: assume the chain is along the specified dimension
  int batch_idx = chain_id / stride;
  int in_idx = chain_id % stride;
  int64_t base = batch_idx * (dim_size * stride) + in_idx;

  // Each thread in the block processes a segment of the chain
  int t = threadIdx.x;
  int T = blockDim.x;
  int chunk = (dim_size + T - 1) / T;  // ceiling division to partition the chain among threads
  int start = t * chunk;
  int end = start + chunk;
  if (end > dim_size) end = dim_size;

  // Declare external shared memory for storing each thread's local product
  extern __shared__ scalar_t s_local[];
  scalar_t local_prod = 1;

  // First pass: each thread computes the product of its assigned segment
  int i = start;
  // Unroll the loop for most iterations
  for (; i + 7 < end; i += 8) {
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      int64_t idx = base + (i + j) * stride;
      local_prod *= input[idx];
    }
  }
  // Handle remaining elements
  for (; i < end; i++) {
    int64_t idx = base + i * stride;
    local_prod *= input[idx];
  }

  // Store each thread's local product in shared memory
  s_local[t] = local_prod;
  __syncthreads();

  // Compute the exclusive product offset for this thread (i.e. product of all segments from threads with lower id)
  scalar_t offset = 1;
  for (int j = 0; j < t; j++) {
    offset *= s_local[j];
  }
  __syncthreads();

  // Second pass: recompute cumulative product for the thread's segment starting from the offset
  scalar_t prod = offset;
  i = start;
  for (; i + 7 < end; i += 8) {
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      int64_t idx = base + (i + j) * stride;
      prod *= input[idx];
      output[idx] = prod;
    }
  }
  for (; i < end; i++) {
    int64_t idx = base + i * stride;
    prod *= input[idx];
    output[idx] = prod;
  }
}

// Forward function that sets up asynchronous transfers and launches the combined kernel

torch::Tensor cumprod_combo_forward(torch::Tensor input, int64_t dim) {
  // Check if the input tensor is on CPU. If so, perform async transfer to GPU.
  bool input_on_cpu = !input.is_cuda();
  torch::Tensor input_device;
  if (input_on_cpu) {
    input_device = input.to(torch::kCUDA, /*non_blocking=*/true);
  } else {
    input_device = input;
  }

  // Allocate output tensor on the same device as input
  auto output_device = torch::empty_like(input_device);

  // Create a non-blocking CUDA stream to overlap kernel execution and memory copies
  cudaStream_t kernel_stream;
  cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking);

  // Retrieve tensor properties
  auto sizes = input_device.sizes();
  auto strides = input_device.strides();
  int64_t dim_size = sizes[dim];
  int64_t stride_val = strides[dim];
  // Each independent cumulative product chain spans the dimension 'dim'
  int64_t total_chains = input_device.numel() / dim_size;

  // Launch configuration: one block per cumulative product chain, 256 threads per block
  int threads = 256;
  dim3 blocks(total_chains);
  dim3 threads_per_block(threads);
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_device.scalar_type(), "cumprod_combo_forward", ([&] {
    // Calculate shared memory size based on number of threads
    size_t shmem_size = threads * sizeof(scalar_t);
    cumprod_combo_kernel<scalar_t><<<blocks, threads_per_block, shmem_size, kernel_stream>>>(
      output_device.data_ptr<scalar_t>(),
      input_device.data_ptr<scalar_t>(),
      dim_size,
      stride_val,
      total_chains
    );
  }));

  torch::Tensor output;
  if (input_on_cpu) {
    // Use pinned memory for asynchronous device-to-host copy
    output = torch::empty_like(output_device,
                                 output_device.options().device(torch::kCPU).pinned_memory(true));
    cudaMemcpyAsync(output.data_ptr(),
                    output_device.data_ptr(),
                    output_device.numel() * output_device.element_size(),
                    cudaMemcpyDeviceToHost,
                    kernel_stream);
  } else {
    output = output_device;
  }

  // Ensure the kernel execution and memory copies are complete
  cudaStreamSynchronize(kernel_stream);
  cudaStreamDestroy(kernel_stream);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cumprod_combo_forward, "Combined efficient cumulative product forward (CUDA) with two-pass scan and streams");
}
