#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel: Computes Sigmoid for a range [start, end) of elements
template <typename scalar_t>
__global__ void sigmoid_kernel_range(const scalar_t * __restrict__ input,
                                       scalar_t * __restrict__ output,
                                       const int64_t start,
                                       const int64_t end) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int64_t i = start + idx; i < end; i += stride) {
    float val = static_cast<float>(input[i]);
    float r = 1.0f / (1.0f + expf(-val));
    output[i] = static_cast<scalar_t>(r);
  }
}

// Host forward function that splits the workload into chunks and uses multiple CUDA streams
// to overlap kernel execution with memory transfers, which can improve pipelining
// when host-device memory copies are involved.

torch::Tensor forward(torch::Tensor input) {
  // Assume input is already on the GPU
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();

  // Determine number of chunks/streams to use
  const int numStreams = 4;
  const int64_t chunkSize = (size + numStreams - 1) / numStreams;

  // Create CUDA streams
  std::vector<cudaStream_t> streams(numStreams);
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  const int threads = 256;

  // Launch a kernel on each chunk in a separate stream
  for (int i = 0; i < numStreams; i++) {
    const int64_t start = i * chunkSize;
    int64_t end = start + chunkSize;
    if (end > size) end = size;
    if (start >= end) break;
    const int64_t chunkElements = end - start;
    const int blocks = (chunkElements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel_range", ([&] {
      sigmoid_kernel_range<scalar_t><<<blocks, threads, 0, streams[i]>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          start,
          end
      );
    }));
  }

  // Synchronize all streams
  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Stream pipelined Sigmoid forward (CUDA)");
}
