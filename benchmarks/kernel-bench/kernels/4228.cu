#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__constant__ float c_min_val;
__constant__ float c_max_val;

__global__ void hardtanh_constant_kernel(const float* __restrict__ x,
                                        float* __restrict__ out,
                                        int64_t numel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = 1;

  for (int i = idx; i < numel; i += stride) {
    const float val = __ldg(x + i);
    out[i] = fminf(fmaxf(val, c_min_val), c_max_val);
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  const int64_t numel = x.numel();

  // Copy min/max to constant memory
  cudaMemcpyToSymbol(c_min_val, &min_val, sizeof(float));
  cudaMemcpyToSymbol(c_max_val, &max_val, sizeof(float));

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", [&] {
    hardtanh_constant_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                                out.data_ptr<float>(),
                                                numel);
  });

  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) throw std::invalid_argument("Input must be CUDA tensor");
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh with constant memory optimization");
}