#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Kernel optimized for overlapping computation and memory transfers
__global__ __launch_bounds__(256) void hardtanh_stream_kernel(const float* __restrict__ x,
                                        float* __restrict__ out,
                                        int64_t numel,
                                        float min_val,
                                        float max_val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    float val = __ldg(&x[idx]);
    out[idx] = fminf(fmaxf(val, min_val), max_val);
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  const int64_t numel = x.numel();
  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  hardtanh_stream_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      numel,
      min_val,
      max_val);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) {
    throw std::invalid_argument("Input tensor must be a CUDA tensor");
  }
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh CUDA optimized with streams");
}