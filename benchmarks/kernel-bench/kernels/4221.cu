#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename T>
__global__ void hardtanh_vectorized_kernel(const T* __restrict__ input,
                                         T* __restrict__ output,
                                         const T min_val,
                                         const T max_val,
                                         int64_t n) {
  const int stride = blockDim.x * gridDim.x;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  while (idx < n) {
    if (idx + 3 < n) {
      float4 vec = __ldg(reinterpret_cast<const float4*>(input + idx));
      vec.x = fminf(fmaxf(vec.x, min_val), max_val);
      vec.y = fminf(fmaxf(vec.y, min_val), max_val);
      vec.z = fminf(fmaxf(vec.z, min_val), max_val);
      vec.w = fminf(fmaxf(vec.w, min_val), max_val);
      *reinterpret_cast<float4*>(output + idx) = vec;
    } else {
      for (int i = idx; i < n && i < idx + 4; ++i) {
        T val = __ldg(input + i);
        output[i] = fminf(fmaxf(val, min_val), max_val);
      }
    }
    idx += stride;
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  const int64_t numel = x.numel();

  const int threads = 256;
  const int blocks = (numel + threads * 4 - 1) / (threads * 4);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", [&] {
    hardtanh_vectorized_kernel<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        static_cast<scalar_t>(min_val),
        static_cast<scalar_t>(max_val),
        numel);
  });

  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) {
    throw std::invalid_argument("Input tensor must be a CUDA tensor");
  }
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh optimized vectorized (CUDA)");
}