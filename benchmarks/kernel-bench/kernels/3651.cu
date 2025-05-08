#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute HardSigmoid on a contiguous chunk of data
// y = clamp((x + 3)/6, 0, 1)
template <typename scalar_t>
__global__ void streamed_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                             scalar_t* __restrict__ output,
                                             size_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  scalar_t add_const = static_cast<scalar_t>(3);
  scalar_t scale = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    scalar_t y = (x + add_const) * scale;
    // Clamp y to the range [0, 1]
    y = (y < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0)
         : (y > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1) : y;
    output[i] = y;
  }
}

// Host function that partitions the input into chunks and processes them on separate CUDA streams

torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  const size_t numel = input.numel();

  // Number of streams to use for overlapping computation and memory operations
  const int nstreams = 4;
  cudaStream_t streams[nstreams];
  for (int i = 0; i < nstreams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // Partition the input into chunks for each stream
  const size_t chunk_size = (numel + nstreams - 1) / nstreams;
  const int threads = 1024;

  for (int i = 0; i < nstreams; i++) {
    size_t offset = i * chunk_size;
    if (offset >= numel) break; // No more data
    size_t current_chunk = (offset + chunk_size > numel) ? (numel - offset) : chunk_size;
    int blocks = (current_chunk + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "streamed_hardsigmoid_cuda", ([&] {
      streamed_hardsigmoid_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
          input.data_ptr<scalar_t>() + offset,
          output.data_ptr<scalar_t>() + offset,
          current_chunk);
    }));
  }

  // Synchronize and destroy streams
  for (int i = 0; i < nstreams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA) with overlapped computation using streams");
}
