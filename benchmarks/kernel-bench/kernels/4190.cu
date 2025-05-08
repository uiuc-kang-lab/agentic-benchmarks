#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Optimized kernel with block size experimentation
template <typename scalar_t>
__global__ void hardtanh_kernel_optimized(const scalar_t* __restrict__ x,
                                           scalar_t* __restrict__ out,
                                           int64_t numel,
                                           scalar_t min_val,
                                           scalar_t max_val) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numel) {
    scalar_t val = x[i];
    // Clamp between min_val and max_val.
    if (val < min_val) {
      val = min_val;
    } else if (val > max_val) {
      val = max_val;
    }
    out[i] = val;
  }
}

at::Tensor forward_cuda_optimized(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  // Experiment with different block sizes
  const int threads = 256;  // Chosen after experimentation
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda_optimized", ([&] {
    hardtanh_kernel_optimized<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        numel,
        static_cast<scalar_t>(min_val),
        static_cast<scalar_t>(max_val)
    );
  }));

  return out;
}

at::Tensor forward_optimized(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) {
    throw std::invalid_argument("Input tensor must be a CUDA tensor");
  }
  return forward_cuda_optimized(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_optimized", &forward_optimized, "HardTanh activation optimized (CUDA)");
}