#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>

// Kernel that processes a chunk of the input tensor starting from a given offset
template <typename scalar_t>
__global__ void hardtanh_kernel_async(const scalar_t* __restrict__ x,
                                        scalar_t* __restrict__ out,
                                        int64_t chunk_num,
                                        int64_t offset,
                                        scalar_t min_val,
                                        scalar_t max_val) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < chunk_num) {
    int64_t idx = i + offset;
    scalar_t val = __ldg(&x[idx]);
    if (val < min_val) {
      val = min_val;
    } else if (val > max_val) {
      val = max_val;
    }
    out[idx] = val;
  }
}

// Host function that partitions the tensor into chunks and pipelines kernel execution using multiple CUDA streams
at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t total_numel = x.numel();
  if (total_numel == 0) return out;

  // Define chunk size. This can be tuned; here we use 1M elements per chunk
  const int64_t CHUNK_SIZE = 1 << 20;  // 1M elements
  int64_t n_chunks = (total_numel + CHUNK_SIZE - 1) / CHUNK_SIZE;

  // Create a fixed number of CUDA streams to enable concurrent kernel execution
  const int num_streams = 4;
  cudaStream_t streams[num_streams];
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  const int threads = 256;

  // Launch kernels for each chunk asynchronously on different streams
  for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
    int64_t offset = chunk * CHUNK_SIZE;
    int64_t current_chunk = std::min(CHUNK_SIZE, total_numel - offset);
    int blocks = (current_chunk + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_kernel_async", ([&] {
      hardtanh_kernel_async<scalar_t><<<blocks, threads, 0, streams[chunk % num_streams]>>>(
          x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),
          current_chunk,
          offset,
          static_cast<scalar_t>(min_val),
          static_cast<scalar_t>(max_val));
    }));
  }

  // Synchronize and destroy all streams
  for (int i = 0; i < num_streams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) {
    throw std::invalid_argument("Input tensor must be a CUDA tensor");
  }
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh activation with pipelined streams (CUDA)");
}
