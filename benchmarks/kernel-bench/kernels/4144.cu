#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

template <typename scalar_t>
__global__ void hardtanh_optimized(const scalar_t* __restrict__ x,
                                  scalar_t* __restrict__ out,
                                  int64_t numel,
                                  scalar_t min_val,
                                  scalar_t max_val) {
  constexpr int VecWidth = (sizeof(scalar_t) == 4) ? 4 : (sizeof(scalar_t) == 8) ? 2 : 1;
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = blockDim.x * gridDim.x;

  // Vectorized processing
  if constexpr (VecWidth > 1) {
    using vec_t = typename std::conditional<sizeof(scalar_t) == 4, float4, 
                          typename std::conditional<sizeof(scalar_t) == 8, double2, scalar_t>::type>::type;
    const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);
    vec_t* out_vec = reinterpret_cast<vec_t*>(out);
    const int64_t vecNum = numel / VecWidth;

    for (int64_t i = tid; i < vecNum; i += stride) {
      vec_t v = __ldg(&x_vec[i]);
      if constexpr (sizeof(scalar_t) == 4) {
        v.x = max(min_val, min(max_val, v.x));
        v.y = max(min_val, min(max_val, v.y));
        v.z = max(min_val, min(max_val, v.z));
        v.w = max(min_val, min(max_val, v.w));
      } else if constexpr (sizeof(scalar_t) == 8) {
        v.x = max(min_val, min(max_val, v.x));
        v.y = max(min_val, min(max_val, v.y));
      }
      out_vec[i] = v;
    }
  }

  // Scalar tail handling
  const int64_t start = (numel / VecWidth) * VecWidth;
  for (int64_t i = start + tid; i < numel; i += stride) {
    const scalar_t val = __ldg(&x[i]);
    out[i] = max(min_val, min(max_val, val));
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  const int64_t numel = x.numel();

  const int threads = 256;
  const int blocks = min((numel + threads - 1) / threads, 65535);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_optimized", ([&] {
    hardtanh_optimized<scalar_t><<<blocks, threads>>>(
      x.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      numel,
      static_cast<scalar_t>(min_val),
      static_cast<scalar_t>(max_val)
    );
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
  m.def("forward", &forward, "HardTanh optimized vectorized (CUDA)");
}
