#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

// Kernel that distributes workloads evenly across threads and blocks
// to avoid underutilization or bottlenecks.
template <typename scalar_t>
__global__ void hardtanh_kernel_balanced(const scalar_t* __restrict__ x,
                                          scalar_t* __restrict__ out,
                                          int64_t numel,
                                          scalar_t min_val,
                                          scalar_t max_val) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total_threads = blockDim.x * gridDim.x;

  for (int64_t i = tid; i < numel; i += total_threads) {
    scalar_t val = x[i];
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
    hardtanh_kernel_balanced<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &forward, "Balanced HardTanh activation (CUDA)");
}