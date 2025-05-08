#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

template <typename scalar_t>
__global__ void hardtanh_shared_vectorized(const scalar_t* __restrict__ x,
                                           scalar_t* __restrict__ out,
                                           int64_t numel,
                                           scalar_t min_val,
                                           scalar_t max_val) {
  constexpr int VEC_WIDTH = (sizeof(scalar_t) == 4) ? 4 : 2;
  using vec_t = typename std::conditional<sizeof(scalar_t) == 4, float4, double2>::type;
  
  extern __shared__ char shared_buffer[];
  vec_t* shared_vec = reinterpret_cast<vec_t*>(shared_buffer);

  const int64_t vec_num = numel / VEC_WIDTH;
  const int64_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t vec_stride = blockDim.x * gridDim.x;

  // Vectorized processing with shared memory
  for (int64_t i = vec_idx; i < vec_num; i += vec_stride) {
    vec_t v = __ldg(reinterpret_cast<const vec_t*>(x) + i);
    shared_vec[threadIdx.x] = v;
    __syncthreads();

    vec_t sv = shared_vec[threadIdx.x];
    if constexpr (sizeof(scalar_t) == 4) {
      sv.x = max(min_val, min(max_val, sv.x));
      sv.y = max(min_val, min(max_val, sv.y));
      sv.z = max(min_val, min(max_val, sv.z));
      sv.w = max(min_val, min(max_val, sv.w));
    } else {
      sv.x = max(min_val, min(max_val, sv.x));
      sv.y = max(min_val, min(max_val, sv.y));
    }

    reinterpret_cast<vec_t*>(out)[i] = sv;
    __syncthreads();
  }

  // Scalar remainder processing
  const int64_t scalar_start = vec_num * VEC_WIDTH;
  const int64_t scalar_idx = scalar_start + blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar_idx < numel) {
    scalar_t val = __ldg(x + scalar_idx);
    out[scalar_idx] = max(min_val, min(max_val, val));
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  const int64_t numel = x.numel();

  constexpr int THREADS = 256;
  const int vec_blocks = (numel / (THREADS * (sizeof(float) == 4 ? 4 : 2))) + 1;
  const int blocks = std::min(vec_blocks, 65535);
  const size_t shared_size = THREADS * sizeof(typename std::conditional<sizeof(float) == 4, float4, double2>::type);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_shared", ([&] {
    hardtanh_shared_vectorized<scalar_t><<<blocks, THREADS, shared_size>>>(
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
  m.def("forward", &forward, "HardTanh with shared memory vectorization (CUDA)");
}