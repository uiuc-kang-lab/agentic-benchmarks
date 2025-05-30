#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Warp-level HardTanh
template <typename scalar_t>
__device__ scalar_t warp_hardtanh_reduce(scalar_t val, scalar_t min_val, scalar_t max_val) {
    // Clamp value using warp primitives
    val = (val < min_val) ? min_val : (val > max_val ? max_val : val);
    return val;
}

// Improved HardTanh kernel using warp-level operations
template <typename scalar_t>
__global__ void hardtanh_warp_kernel(const scalar_t* __restrict__ x,
                                     scalar_t* __restrict__ out,
                                     int64_t numel,
                                     scalar_t min_val,
                                     scalar_t max_val) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numel) {
    scalar_t val = x[i];
    val = warp_hardtanh_reduce(val, min_val, max_val);
    out[i] = val;
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
    hardtanh_warp_kernel<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &forward, "HardTanh activation warp-optimized (CUDA)");
}