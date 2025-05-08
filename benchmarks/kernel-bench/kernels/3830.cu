#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel that processes a chunk of the input starting from a given offset
template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int chunk_size,
    const int offset) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < chunk_size) {
    int global_idx = idx + offset;
    const scalar_t x = input[global_idx];
    if (x > scalar_t(20.0)) {
      output[global_idx] = x;
    } else if (x < scalar_t(-20.0)) {
      output[global_idx] = exp(x);
    } else {
      output[global_idx] = log1p(exp(x));
    }
  }
}

// This function splits the input tensor into several chunks and processes each chunk
// on its own CUDA stream. The idea is to overlap memory operations (loads/stores)
// with computation by pipelining execution across streams, which can reduce overall runtime
// on large tensors, especially on hardware like the H100 with CUDA 12.2.

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

  auto output = torch::empty_like(input);
  const int total_elements = input.numel();

  // Define number of streams and chunk size to split the work
  const int num_streams = 4;
  const int chunk_size = (total_elements + num_streams - 1) / num_streams;

  // Create CUDA streams
  std::vector<cudaStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  const int threads = 256;
  // Launch the kernel for each chunk on its own stream
  for (int i = 0; i < num_streams; i++) {
    int offset = i * chunk_size;
    if (offset >= total_elements) break;
    int current_chunk_size = std::min(chunk_size, total_elements - offset);
    int blocks = (current_chunk_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda_stream", ([&] {
      softplus_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          current_chunk_size,
          offset);
    }));
  }

  // Synchronize and destroy streams
  for (int i = 0; i < num_streams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &softplus_cuda_forward, "Softplus forward with stream pipeline (CUDA)");
}
