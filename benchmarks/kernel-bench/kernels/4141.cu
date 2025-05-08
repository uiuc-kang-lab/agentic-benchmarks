#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

template <typename scalar_t>
__global__ void hardtanh_kernel_combined(const scalar_t* __restrict__ x,
                                         scalar_t* __restrict__ out,
                                         int64_t numel,
                                         scalar_t min_val,
                                         scalar_t max_val) {
  constexpr int VecWidth = (sizeof(scalar_t) == 4 ? 4 : (sizeof(scalar_t) == 8 ? 2 : 1));
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  int clamped_flag = 0;

  // Vectorized processing
  if constexpr (VecWidth > 1) {
    using vec_t = typename std::conditional<sizeof(scalar_t) == 4, float4,
                    typename std::conditional<sizeof(scalar_t) == 8, double2, scalar_t>::type>::type;
    int64_t vecNum = numel / VecWidth;
    const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);
    vec_t* out_vec = reinterpret_cast<vec_t*>(out);

    for (int64_t i = tid; i < vecNum; i += stride) {
      vec_t v = __ldg(&x_vec[i]);
      bool any_clamped = false;

      if constexpr (sizeof(scalar_t) == 4) {
        if (v.x < min_val) { v.x = min_val; any_clamped = true; }
        else if (v.x > max_val) { v.x = max_val; any_clamped = true; }
        if (v.y < min_val) { v.y = min_val; any_clamped = true; }
        else if (v.y > max_val) { v.y = max_val; any_clamped = true; }
        if (v.z < min_val) { v.z = min_val; any_clamped = true; }
        else if (v.z > max_val) { v.z = max_val; any_clamped = true; }
        if (v.w < min_val) { v.w = min_val; any_clamped = true; }
        else if (v.w > max_val) { v.w = max_val; any_clamped = true; }
      } else if (sizeof(scalar_t) == 8) {
        if (v.x < min_val) { v.x = min_val; any_clamped = true; }
        else if (v.x > max_val) { v.x = max_val; any_clamped = true; }
        if (v.y < min_val) { v.y = min_val; any_clamped = true; }
        else if (v.y > max_val) { v.y = max_val; any_clamped = true; }
      }

      if (any_clamped) clamped_flag = 1;
      out_vec[i] = v;
    }
  }

  // Scalar processing for remaining elements
  int64_t start = (numel / VecWidth) * VecWidth;
  for (int64_t i = start + tid; i < numel; i += stride) {
    scalar_t val = __ldg(&x[i]);
    if (val < min_val) {
      out[i] = min_val;
      clamped_flag = 1;
    } else if (val > max_val) {
      out[i] = max_val;
      clamped_flag = 1;
    } else {
      out[i] = val;
    }
  }

  // Warp-level reduction
  unsigned mask = 0xffffffff;
  int warp_total = clamped_flag;
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    warp_total += __shfl_down_sync(mask, warp_total, offset);
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_combined", ([&] {
    hardtanh_kernel_combined<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &forward, "HardTanh with vectorized loads + warp reduction (CUDA)");
}
