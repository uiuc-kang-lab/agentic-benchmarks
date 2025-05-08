#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename scalar_t>
__global__ void hardtanh_grid_stride_kernel(const scalar_t* __restrict__ x,
                                          scalar_t* __restrict__ out,
                                          int64_t numel,
                                          scalar_t min_val,
                                          scalar_t max_val) {
  const int64_t grid_stride = blockDim.x * gridDim.x;
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  
  while (i < numel) {
    scalar_t val = x[i];
    out[i] = max(min(val, max_val), min_val);
    i += grid_stride;
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 256;
  int blocks = (numel + threads - 1) / threads;
  auto* device_properties = at::cuda::getCurrentDeviceProperties();
  blocks = std::min(blocks, device_properties->maxGridSize[0]);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_grid_stride_cuda", ([&] {
    hardtanh_grid_stride_kernel<scalar_t><<<blocks, threads>>>(
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