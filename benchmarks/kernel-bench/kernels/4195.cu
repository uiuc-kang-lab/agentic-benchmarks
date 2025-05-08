#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename scalar_t>
__global__ void hardtanh_kernel_unrolled(const scalar_t* __restrict__ x,
                                scalar_t* __restrict__ out,
                                int64_t numel,
                                scalar_t min_val,
                                scalar_t max_val) {
  constexpr int UNROLL_FACTOR = 4;
  int64_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * UNROLL_FACTOR;

  #pragma unroll
  for (int j = 0; j < UNROLL_FACTOR; j++) {
    int64_t i = base_idx + j;
    if (i < numel) {
      // Use read-only cache for load
      scalar_t val = __ldg(&x[i]);
      // Compute HardTanh using inline conditional operations to potentially reduce register pressure
      out[i] = (val > max_val) ? max_val : ((val < min_val) ? min_val : val);
    }
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 512;
  const int elements_per_block = threads * 4;
  const int blocks = (numel + elements_per_block - 1) / elements_per_block;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
    hardtanh_kernel_unrolled<scalar_t><<<blocks, threads>>>(
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
