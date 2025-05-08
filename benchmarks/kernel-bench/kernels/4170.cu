#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel applying HardTanh activation using a grid-stride loop
// No unnecessary __syncthreads() are used as each thread operates independently.

template <typename scalar_t>
__global__ void hardtanh_grid_stride_kernel(const scalar_t* __restrict__ x,
                                              scalar_t* __restrict__ out,
                                              int64_t numel,
                                              scalar_t min_val,
                                              scalar_t max_val) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (; index < numel; index += stride) {
    scalar_t val = x[index];
    // Branchless clamping: clamp val between min_val and max_val
    val = (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
    out[index] = val;
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  // Calculate optimal grid dimensions based on device properties
  const int threads = 256;  // Reduced thread count for better occupancy
  const int max_blocks = 65535;  // Maximum blocks per grid dimension
  // Calculate blocks based on input size and device capabilities
  const int blocks = std::min(max_blocks, (numel + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
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
  m.def("forward", &forward, "HardTanh activation grid-stride kernel (CUDA) without unnecessary synchronizations");
}
