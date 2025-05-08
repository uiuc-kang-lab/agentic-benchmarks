#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Simple elementwise tanh kernel processing a contiguous chunk.
template <typename scalar_t>
__global__ void tanh_kernel_overlap(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = tanhf(input[idx]);
  }
}

// Forward function that splits the work into chunks and uses multiple CUDA streams
// to overlap a device-to-device copy (simulating a memory transfer) with kernel computation.
// This pipelining can help hide memory copy latency by concurrently copying data from
// the original input to a temporary buffer while computing on previously copied data.

torch::Tensor forward(torch::Tensor input) {
  // Ensure that the input is on CUDA
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

  // Allocate output tensor (same type and shape as input)
  auto output = torch::empty_like(input);
  // Allocate a temporary buffer. In a real pipelining scenario, this could be used to
  // overlap host-device transfers. Here we simulate a memory transfer by doing a device-to-device copy.
  auto temp = torch::empty_like(input);

  int64_t size = input.numel();
  // Set a chunk size (number of elements per chunk)
  const int chunk_size = 1 << 16; // 65536 elements per chunk (tunable)
  int num_chunks = (size + chunk_size - 1) / chunk_size;

  // Allocate an array of CUDA streams, one per chunk
  cudaStream_t* streams = new cudaStream_t[num_chunks];
  for (int i = 0; i < num_chunks; i++) {
    cudaStreamCreate(&streams[i]);
  }

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "stream_overlap_tanh_kernel", ([&] {
    const scalar_t* in_ptr = input.data_ptr<scalar_t>();
    scalar_t* out_ptr = output.data_ptr<scalar_t>();
    scalar_t* temp_ptr = temp.data_ptr<scalar_t>();

    for (int i = 0; i < num_chunks; i++) {
      int64_t start = i * chunk_size;
      int64_t end = std::min(start + (int64_t)chunk_size, size);
      int current_chunk = end - start;
      size_t bytes = current_chunk * sizeof(scalar_t);

      // Asynchronously copy the chunk from input to temporary buffer using the i-th stream
      cudaMemcpyAsync(temp_ptr + start, in_ptr + start, bytes, cudaMemcpyDeviceToDevice, streams[i]);

      // Launch the tanh kernel for this chunk on the same stream
      int threads = 256;
      int blocks = (current_chunk + threads - 1) / threads;
      tanh_kernel_overlap<scalar_t><<<blocks, threads, 0, streams[i]>>>(
          temp_ptr + start, out_ptr + start, current_chunk);
    }
  }));

  // Synchronize and destroy streams 
  for (int i = 0; i < num_chunks; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }
  delete[] streams;

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Tanh forward with stream overlapping (CUDA)");
}
