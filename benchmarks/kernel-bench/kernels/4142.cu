#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

template <typename scalar_t>
__global__ void hardtanh_coalesced(const scalar_t* __restrict__ x,
                                  scalar_t* __restrict__ out,
                                  int64_t numel,
                                  scalar_t min_val,
                                  scalar_t max_val) {
  constexpr int VecWidth = (sizeof(scalar_t) == 4 ? 4 : 2);
  const int64_t vec_numel = numel / VecWidth;
  
  // Process aligned vector elements
  const int64_t block_vec_start = blockIdx.x * blockDim.x * VecWidth;
  const int64_t thread_vec_offset = threadIdx.x * VecWidth;
  
  using vec_t = typename std::conditional<sizeof(scalar_t) == 4, float4, double2>::type;
  const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);
  vec_t* out_vec = reinterpret_cast<vec_t*>(out);

  // Process full vectors in coalesced pattern
  for (int64_t vidx = block_vec_start + threadIdx.x; 
       vidx < vec_numel; 
       vidx += blockDim.x * gridDim.x) {
    vec_t v = __ldg(&x_vec[vidx]);
    
    if constexpr (sizeof(scalar_t) == 4) {
      v.x = fminf(fmaxf(v.x, min_val), max_val);
      v.y = fminf(fmaxf(v.y, min_val), max_val);
      v.z = fminf(fmaxf(v.z, min_val), max_val);
      v.w = fminf(fmaxf(v.w, min_val), max_val);
    } else {
      v.x = fmin(fmax(v.x, min_val), max_val);
      v.y = fmin(fmax(v.y, min_val), max_val);
    }
    out_vec[vidx] = v;
  }

  // Process remaining elements
  const int64_t scalar_start = vec_numel * VecWidth;
  const int64_t global_idx = block_vec_start + thread_vec_offset + scalar_start;
  
  for (int64_t i = global_idx; i < numel; i += blockDim.x * gridDim.x * VecWidth) {
    if (i < numel) {
      scalar_t val = __ldg(x + i);
      out[i] = fmin(fmax(val, min_val), max_val);
    }
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  const int64_t numel = x.numel();

  constexpr int threads = 128;  // 4 warps per block
  const int blocks = (numel + (threads * 4) - 1) / (threads * 4);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_coalesced", ([&] {
    hardtanh_coalesced<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &forward, "HardTanh with coalesced memory access (CUDA)");
}
