#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename scalar_t>
__global__ void hardtanh_scalar_ldg(const scalar_t* __restrict__ x,
                                   scalar_t* out,
                                   int64_t numel,
                                   scalar_t min_val,
                                   scalar_t max_val) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numel) {
    out[i] = fminf(fmaxf(__ldg(x + i), min_val), max_val);
  }
}

__global__ void hardtanh_float4_aligned(const float* __restrict__ x,
                                       float* out,
                                       int64_t numel,
                                       float min,
                                       float max) {
  constexpr int VEC_SIZE = 4;
  
  // Aligned base pointer for vector loads
  const float4* x_vec = reinterpret_cast<const float4*>(x);
  float4* out_vec = reinterpret_cast<float4*>(out);

  // Process full vectors
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx * VEC_SIZE < numel - (VEC_SIZE - 1)) {
    float4 vec = __ldg(&x_vec[idx]);
    vec.x = fminf(fmaxf(vec.x, min), max);
    vec.y = fminf(fmaxf(vec.y, min), max);
    vec.z = fminf(fmaxf(vec.z, min), max);
    vec.w = fminf(fmaxf(vec.w, min), max);
    out_vec[idx] = vec;
    return;
  }

  // Handle remaining elements with scalar operations
  idx = idx * VEC_SIZE;
  for (int i = 0; i < VEC_SIZE && idx + i < numel; ++i) {
    out[idx + i] = fminf(fmaxf(__ldg(x + idx + i), min), max);
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  const int64_t numel = x.numel();
  
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", [&] {
    if (std::is_same<scalar_t, float>::value && numel % 4 == 0) {
      constexpr int VEC_SIZE = 4;
      const int threads = 256;
      const int vec_count = numel / VEC_SIZE;
      const int blocks = (vec_count + threads - 1) / threads;
      hardtanh_float4_aligned<<<blocks, threads>>>(x.data_ptr<float>(),
                                                  out.data_ptr<float>(),
                                                  numel,
                                                  min_val,
                                                  max_val);
    } else {
      const int threads = 512;
      const int blocks = (numel + threads - 1) / threads;
      hardtanh_scalar_ldg<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                        out.data_ptr<scalar_t>(),
                                                        numel,
                                                        min_val,
                                                        max_val);
    }
  });

  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) throw std::invalid_argument("Input must be CUDA tensor");
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh optimized with vector loads");
}