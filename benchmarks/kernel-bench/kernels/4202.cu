#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Use better indexing with grid-stride loop
__global__ void hardtanh_optimized_kernel(const float* __restrict__ x,
                                           float* __restrict__ out,
                                           int64_t numel,
                                           float min_val,
                                           float max_val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numel; i += stride) {
    float val = x[i];
    out[i] = fminf(fmaxf(val, min_val), max_val);
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  hardtanh_optimized_kernel<<<blocks, threads>>>(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      numel,
      min_val,
      max_val);

  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) {
    throw std::invalid_argument("Input tensor must be a CUDA tensor");
  }
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh optimized (CUDA)");
}