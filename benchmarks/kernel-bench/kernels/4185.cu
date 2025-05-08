#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

// This kernel uses vectorized loads/stores with __ldg to optimize global memory accesses
// by aligning them to 128-bit boundaries. For floats, we use float4 (128 bits = 4 floats),
// and for doubles we use double2 (128 bits = 2 doubles). Tail elements are processed separately.

template <typename scalar_t, typename vector_t, int VecSize>
__global__ void hardtanh_vectorized_kernel(const scalar_t* __restrict__ x,
                                             scalar_t* __restrict__ out,
                                             int64_t numel,
                                             scalar_t min_val,
                                             scalar_t max_val) {
  // Number of vectorized elements
  int64_t num_vec = numel / VecSize;
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  // Union to reinterpret the vector load as an array of scalars
  union {
    vector_t vec;
    scalar_t s[VecSize];
  } data;

  // Process the bulk of the data with 128-bit aligned vectorized loads
  for (int64_t vec_index = index; vec_index < num_vec; vec_index += stride) {
    // Use __ldg for read-only access
    vector_t v = __ldg(reinterpret_cast<const vector_t*>(x) + vec_index);
    data.vec = v;
    #pragma unroll
    for (int i = 0; i < VecSize; i++) {
      scalar_t elem = data.s[i];
      // Apply HardTanh activation
      elem = (elem < min_val) ? min_val : ((elem > max_val) ? max_val : elem);
      data.s[i] = elem;
    }
    reinterpret_cast<vector_t*>(out)[vec_index] = data.vec;
  }

  // Process any remaining tail elements that don't fit into a full vector load
  int64_t tail_start = num_vec * VecSize;
  for (int64_t i = tail_start + index; i < numel; i += stride) {
    scalar_t elem = __ldg(x + i);
    elem = (elem < min_val) ? min_val : ((elem > max_val) ? max_val : elem);
    out[i] = elem;
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();
  const int threads = 1024;
  int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_vectorized_cuda", ([&] {
    if (std::is_same<scalar_t, float>::value) {
      // For float, use float4: 4 floats = 128 bits
      constexpr int vecSize = 4;
      int64_t num_vec = numel / vecSize;
      blocks = (num_vec + threads - 1) / threads;
      hardtanh_vectorized_kernel<scalar_t, float4, vecSize><<<blocks, threads>>>(
          x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),
          numel,
          static_cast<scalar_t>(min_val),
          static_cast<scalar_t>(max_val)
      );
    } else if (std::is_same<scalar_t, double>::value) {
      // For double, use double2: 2 doubles = 128 bits
      constexpr int vecSize = 2;
      int64_t num_vec = numel / vecSize;
      blocks = (num_vec + threads - 1) / threads;
      hardtanh_vectorized_kernel<scalar_t, double2, vecSize><<<blocks, threads>>>(
          x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),
          numel,
          static_cast<scalar_t>(min_val),
          static_cast<scalar_t>(max_val)
      );
    }
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
  m.def("forward", &forward, "Vectorized HardTanh activation (CUDA) with __ldg and 128-bit alignment");
}
