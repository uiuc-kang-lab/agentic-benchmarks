#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Kernel optimized by minimizing atomic operations
__global__ void hardtanh_atomic_minimized_kernel(const float* __restrict__ x,
                                                  float* __restrict__ out,
                                                  int64_t numel,
                                                  float min_val,
                                                  float max_val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    float val = __ldg(&x[idx]);
    // Minimize atomic usage by directly assigning clamped values
    out[idx] = fminf(fmaxf(val, min_val), max_val);
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
    hardtanh_atomic_minimized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        min_val,
        max_val
    );
  }));

  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) throw std::invalid_argument("Input tensor must be CUDA");
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh CUDA optimized");
}
