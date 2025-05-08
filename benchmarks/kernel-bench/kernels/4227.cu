#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Kernel optimized using warp-level primitives
__global__ void hardtanh_warp_optimized_kernel(const float* __restrict__ x,
                                                float* __restrict__ out,
                                                int64_t numel,
                                                float min_val,
                                                float max_val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // int lane = threadIdx.x % 32; // Warp lane index (not used)

  if (idx < numel) {
    float val = __ldg(&x[idx]);
    val = fminf(fmaxf(val, min_val), max_val);

    // Use warp shuffle to broadcast the min and max values across the warp
    float min_val_warp = __shfl_sync(0xFFFFFFFF, min_val, 0);
    float max_val_warp = __shfl_sync(0xFFFFFFFF, max_val, 0);

    // Ensure all threads in the warp have the same min and max values
    out[idx] = fminf(fmaxf(val, min_val_warp), max_val_warp);
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  hardtanh_warp_optimized_kernel<<<blocks, threads>>>(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      numel,
      min_val,
      max_val);

  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) {
    throw std::invalid_argument("Input tensor must be a CUDA tensor");
  }
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh warp optimized (CUDA)");
}