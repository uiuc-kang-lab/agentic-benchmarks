#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Device function to perform HardTanh clamp
template <typename T>
__device__ __forceinline__ T hardtanh_val(const T x, const T min_val, const T max_val) {
  return (x < min_val) ? min_val : ((x > max_val) ? max_val : x);
}

// Modular kernel using a grid-stride loop and the device clamp function
template <typename scalar_t>
__global__ void hardtanh_kernel_modular(const scalar_t* __restrict__ x,
                                          scalar_t* __restrict__ out,
                                          int64_t numel,
                                          const scalar_t min_val,
                                          const scalar_t max_val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < numel; i += stride) {
    // Use __ldg for efficient load from global memory
    scalar_t val = __ldg(&x[i]);
    scalar_t clamped_val = hardtanh_val(val, min_val, max_val); out[i] = clamped_val;
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();
  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda_modular", ([&] {
    hardtanh_kernel_modular<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        numel,
        static_cast<scalar_t>(min_val),
        static_cast<scalar_t>(max_val));
  }));

  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) {
    throw std::invalid_argument("Input tensor must be a CUDA tensor");
  }
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Modular HardTanh activation (CUDA)");
}
