#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <c10/cuda/CUDAStream.h>

// Kernel: Each warp (32 threads) computes the min reduction for one output element.
// The input is logically reshaped as [outer, r, inner] and the reduction is performed along the r dimension.

template <typename scalar_t>
__global__ void min_reduce_fused_warp_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  const int warpSize = 32;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = idx / warpSize;
  if (warp_id >= outer * inner) return;

  int lane = threadIdx.x % warpSize;
  int outer_idx = warp_id / inner;
  int inner_idx = warp_id % inner;
  int base = outer_idx * (r * inner) + inner_idx;

  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  // Each thread in the warp processes elements from the r dimension in strides of warpSize
  for (int j = lane; j < r; j += warpSize) {
    scalar_t val = input[base + j * inner];
    local_min = (val < local_min) ? val : local_min;
  }

  // Warp-level reduction using shuffle down
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, local_min, offset);
    local_min = (other < local_min) ? other : local_min;
  }

  if (lane == 0) {
    output[warp_id] = local_min;
  }
}


// Forward function: Implements pipelining by splitting the work into chunks along the 'outer' dimension
// and overlapping kernel execution with asynchronous device-to-host memory copies using two CUDA streams.
// The final result is returned as a CPU tensor allocated in pinned memory.

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute dimensions: outer (dimensions before 'dim'), r (reduction dim), inner (dimensions after 'dim')
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Construct the final output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }

  // Allocate a device tensor for the complete output
  auto options = input.options();
  torch::Tensor output_device = torch::empty(output_shape, options);

  // Allocate pinned host memory for the final result to enable asynchronous memcpy
  auto host_options = torch::TensorOptions()
                           .dtype(input.dtype())
                           .device(torch::kCPU)
                           .pinned_memory(true);
  torch::Tensor output_host = torch::empty(output_shape, host_options);

  // Determine chunking along the 'outer' dimension to enable pipelining
  int chunk_size = 64; // Tunable: process 'chunk_size' outer elements per iteration
  if (chunk_size > outer) chunk_size = outer;
  int num_chunks = (outer + chunk_size - 1) / chunk_size;

  // Create two non-blocking CUDA streams for overlapping kernel execution and memcpy
  cudaStream_t stream0, stream1;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);

  // Get raw pointers to the input and output data
  auto input_ptr = input.data_ptr();
  auto output_device_ptr = output_device.data_ptr();

  // Process each chunk
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    int current_outer = std::min(chunk_size, outer - chunk * chunk_size);
    int total_output = current_outer * inner; // number of output elements in this chunk

    // Each output element is computed by a warp (32 threads)
    int total_threads = total_output * 32;
    int threads_per_block = 128;  // Tunable block size
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // Compute pointer offsets (in number of elements)
    int input_offset = chunk * chunk_size * r * inner;
    int output_offset = chunk * chunk_size * inner;

    // Launch the kernel for this chunk on one of the streams (ping-pong)
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_async_pipeline", ([&] {
      using scalar_t = scalar;
      const scalar_t* input_chunk = ((const scalar_t*)input_ptr) + input_offset;
      scalar_t* output_chunk_device = ((scalar_t*)output_device_ptr) + output_offset;
      cudaStream_t stream = (chunk % 2 == 0) ? stream0 : stream1;
      min_reduce_fused_warp_kernel<scalar_t><<<num_blocks, threads_per_block, 0, stream>>>(
          input_chunk,
          output_chunk_device,
          current_outer, // chunk's outer dimension
          r,
          inner);
    }));

    // Asynchronously copy the computed chunk from device to pinned host memory
    size_t copy_bytes = static_cast<size_t>(current_outer * inner) * input.element_size();
    const void* src_ptr = output_device.data_ptr();
    void* dst_ptr = output_host.data_ptr();
    src_ptr = static_cast<const char*>(src_ptr) + output_offset * input.element_size();
    dst_ptr = static_cast<char*>(dst_ptr) + output_offset * input.element_size();
    cudaStream_t stream = (chunk % 2 == 0) ? stream0 : stream1;
    cudaMemcpyAsync(dst_ptr, src_ptr, copy_bytes, cudaMemcpyDeviceToHost, stream);
  }

  // Synchronize streams to ensure all operations complete
  cudaStreamSynchronize(stream0);
  cudaStreamSynchronize(stream1);

  cudaStreamDestroy(stream0);
  cudaStreamDestroy(stream1);

  // Return the final result from the pinned host memory.
  // This pipelined approach overlaps kernel execution with memory transfers, reducing overall runtime.
  return output_host;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction with asynchronous pipelining (CUDA)");
}
