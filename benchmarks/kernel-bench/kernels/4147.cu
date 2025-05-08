#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

// Optimized HardTanh kernel using __ldg() for read-only accesses and
// vectorized loads/stores on 128-bit boundaries (float4 for float, double2 for double).
// This kernel processes multiple elements per thread to minimize the number of global
// memory transactions and improve throughput.

template <typename scalar_t>
__global__ void hardtanh_kernel_optimized(const scalar_t* __restrict__ x,
                                            scalar_t* __restrict__ out,
                                            int64_t numel,
                                            scalar_t min_val,
                                            scalar_t max_val) {
  // Determine vector width. For float, we use 4 elements (16 bytes), for double, 2 elements (16 bytes).
  constexpr int VecWidth = (sizeof(scalar_t) == 4 ? 4 : (sizeof(scalar_t) == 8 ? 2 : 1));
  using vec_t = typename std::conditional<sizeof(scalar_t) == 4, float4, double2>::type;

  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  int64_t vec_num = numel / VecWidth;  // Number of vectorized elements

  // Process the bulk of data using vectorized loads and stores
  for (int64_t i = tid; i < vec_num; i += stride) {
    // Load vectorized data using __ldg() for read-only caching
    vec_t v = __ldg(reinterpret_cast<const vec_t*>(x) + i);

    // Clamp each element in the vector
    if constexpr (sizeof(scalar_t) == 4) {
      v.x = v.x < min_val ? min_val : (v.x > max_val ? max_val : v.x);
      v.y = v.y < min_val ? min_val : (v.y > max_val ? max_val : v.y);
      v.z = v.z < min_val ? min_val : (v.z > max_val ? max_val : v.z);
      v.w = v.w < min_val ? min_val : (v.w > max_val ? max_val : v.w);
    } else {
      v.x = v.x < min_val ? min_val : (v.x > max_val ? max_val : v.x);
      v.y = v.y < min_val ? min_val : (v.y > max_val ? max_val : v.y);
    }

    // Store the result back
    reinterpret_cast<vec_t*>(out)[i] = v;
  }

  // Process any remaining scalars
  int64_t scalar_start = vec_num * VecWidth;
  for (int64_t i = scalar_start + tid; i < numel; i += stride) {
    scalar_t v = __ldg(x + i);
    v = v < min_val ? min_val : (v > max_val ? max_val : v);
    out[i] = v;
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_optimized", ([&] {
      hardtanh_kernel_optimized<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &forward, "Optimized HardTanh activation with vectorized global memory access (CUDA)");
}
