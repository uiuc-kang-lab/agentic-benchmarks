#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

// This kernel uses vectorized loads/stores with __ldg() to optimize global memory accesses.
// For float (4 bytes) we use float4 (128-bit) and for double (8 bytes) we use double2 (128-bit).
// This ensures that memory accesses are aligned on 128-bit boundaries, reducing memory transactions.

template <typename scalar_t>
__global__ void hardtanh_kernel_vectorized(const scalar_t* __restrict__ x,
                                             scalar_t* __restrict__ out,
                                             int64_t numel,
                                             scalar_t min_val,
                                             scalar_t max_val) {
  // Determine the vector width: number of scalar_t elements that fit in 128 bits
  constexpr int VecWidth = (sizeof(scalar_t) == 4 ? 4 : (sizeof(scalar_t) == 8 ? 2 : 1));
  
  // Global thread index and stride
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  
  // Number of vectorized loads possible
  int64_t vecNum = numel / VecWidth;

  // Use vectorized loads/stores if possible
  if constexpr (VecWidth > 1) {
    // Choose the appropriate vector type
    using vec_t = typename std::conditional<sizeof(scalar_t) == 4, float4,
                    typename std::conditional<sizeof(scalar_t) == 8, double2, scalar_t>::type>::type;
    const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);
    vec_t* out_vec = reinterpret_cast<vec_t*>(out);

    for (int64_t i = tid; i < vecNum; i += stride) {
      vec_t v = __ldg(&x_vec[i]);
      // Process each element in the vector
      if constexpr (sizeof(scalar_t) == 4) {
        // For float4
        v.x = v.x < min_val ? min_val : (v.x > max_val ? max_val : v.x);
        v.y = v.y < min_val ? min_val : (v.y > max_val ? max_val : v.y);
        v.z = v.z < min_val ? min_val : (v.z > max_val ? max_val : v.z);
        v.w = v.w < min_val ? min_val : (v.w > max_val ? max_val : v.w);
      } else {
        // For double2
        v.x = v.x < min_val ? min_val : (v.x > max_val ? max_val : v.x);
        v.y = v.y < min_val ? min_val : (v.y > max_val ? max_val : v.y);
      }
      out_vec[i] = v;
    }
  }

  // Process remaining elements that weren't handled by vectorized loads
  int64_t start = vecNum * VecWidth;
  for (int64_t i = start + tid; i < numel; i += stride) {
    scalar_t val = __ldg(&x[i]);
    out[i] = val < min_val ? min_val : (val > max_val ? max_val : val);
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535;
    
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
    hardtanh_kernel_vectorized<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &forward, "HardTanh activation (CUDA)");
}
