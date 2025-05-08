#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__constant__ float const_min_val;
__constant__ float const_max_val;

template <typename scalar_t>
__global__ void hardtanh_kernel_constant(const scalar_t* __restrict__ x,
                                          scalar_t* __restrict__ out,
                                          int64_t numel) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  while (i < numel) {
    scalar_t val = x[i];
    // Clamp between const_min_val and const_max_val.
    if (val < const_min_val) {
      val = const_min_val;
    } else if (val > const_max_val) {
      val = const_max_val;
    }
    out[i] = val;
    i += stride;
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  cudaMemcpyToSymbol(const_min_val, &min_val, sizeof(float));
  cudaMemcpyToSymbol(const_max_val, &max_val, sizeof(float));

  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
    hardtanh_kernel_constant<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        numel
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