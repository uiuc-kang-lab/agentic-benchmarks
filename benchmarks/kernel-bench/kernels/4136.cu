#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// This kernel applies the HardTanh activation element-wise and demonstrates 
// the use of warp-level primitives (using __shfl_down_sync) to perform a reduction 
// of a per-thread clamping flag. This reduction can be used to get statistics (e.g., 
// count of clamped elements) without resorting to shared memory.

template <typename scalar_t>
__global__ void hardtanh_kernel_warp(const scalar_t* __restrict__ x,
                                       scalar_t* __restrict__ out,
                                       int64_t numel,
                                       scalar_t min_val,
                                       scalar_t max_val) {
  // Global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread will set this flag if it clamps any value
  int clamped_flag = 0;
  
  // Use a grid-stride loop so that all elements are processed
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < numel; i += stride) {
    scalar_t val = x[i];
    scalar_t res;
    if (val < min_val) {
      res = min_val;
      clamped_flag = 1;  // mark that a clamping occurred
    } else if (val > max_val) {
      res = max_val;
      clamped_flag = 1;
    } else {
      res = val;
    }
    out[i] = res;
  }
  
  // Demonstrate warp-level reduction using __shfl_down_sync to sum clamped flags
  // This replaces a potential shared memory reduction for small reductions.
  unsigned mask = 0xffffffff; // Full mask for a warp (32 threads)
  int warp_total = clamped_flag;

  // Perform reduction within the warp
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    warp_total += __shfl_down_sync(mask, warp_total, offset);
  }

  // Optionally, the first thread in each warp (lane 0) could write the warp_total 
  // to a global memory buffer for further aggregation. Here, we simply demonstrate 
  // the warp-level primitive and do not use warp_total further to avoid overhead.
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();
  
  // Launch configuration: using 256 threads per block
  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda_warp", ([&] {
    hardtanh_kernel_warp<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &forward, "HardTanh activation with warp-level reduction (CUDA)");
}
