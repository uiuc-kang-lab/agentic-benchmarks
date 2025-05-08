#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename scalar_t>
__global__ void hardtanh_scalar_kernel(const scalar_t* __restrict__ x,
                                      scalar_t* __restrict__ out,
                                      int64_t numel,
                                      scalar_t min_val,
                                      scalar_t max_val) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numel) {
    scalar_t val = __ldg(x + i);
    out[i] = val < min_val ? min_val : (val > max_val ? max_val : val);
  }
}

__global__ void hardtanh_float4_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      int64_t numel,
                                      float min_val,
                                      float max_val) {
  const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  
  if (idx >= numel) return;

  int64_t remain = numel - idx;
  if (remain >= 4) {
    float4 vec = __ldg(reinterpret_cast<const float4*>(x + idx));
    vec.x = fminf(fmaxf(vec.x, min_val), max_val);
    vec.y = fminf(fmaxf(vec.y, min_val), max_val);
    vec.z = fminf(fmaxf(vec.z, min_val), max_val);
    vec.w = fminf(fmaxf(vec.w, min_val), max_val);
    *reinterpret_cast<float4*>(out + idx) = vec;
  } else {
    for (int64_t i = idx; i < numel; ++i) {
      float val = __ldg(x + i);
      out[i] = fminf(fmaxf(val, min_val), max_val);
    }
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  const int64_t numel = x.numel();
  const int threads = 1024;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", [&] {
    if (std::is_same<scalar_t, float>::value) {
      int64_t items = (numel + 3) / 4;
      int64_t blocks = (items + threads - 1) / threads;
      hardtanh_float4_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                                 out.data_ptr<float>(),
                                                 numel,
                                                 min_val,
                                                 max_val);
    } else {
      int64_t blocks = (numel + threads - 1) / threads;
      hardtanh_scalar_kernel<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                           out.data_ptr<scalar_t>(),
                                                           numel,
                                                           min_val,
                                                           max_val);
    }
  });
  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) throw std::invalid_argument("Input tensor must be CUDA");
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh CUDA optimized");
}