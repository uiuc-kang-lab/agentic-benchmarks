#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

// This kernel uses optimized indexing to ensure efficient use of threads and memory coalescing.
template <typename scalar_t>
__global__ void hardtanh_kernel_1d_optimized(const scalar_t* __restrict__ x,
                                             scalar_t* __restrict__ out,
                                             int64_t numel,
                                             scalar_t min_val,
                                             scalar_t max_val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; idx < numel; idx += stride) {
    scalar_t val = x[idx];
    // Use a single read-modify-write approach for each value
    out[idx] = val < min_val ? min_val : (val > max_val ? max_val : val);
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 512;  // Optimized block size for occupancy and performance
  const int blocks = (numel + threads - 1) / threads;
  
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
    hardtanh_kernel_1d_optimized<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &forward, "Efficient HardTanh activation (CUDA)");
}
