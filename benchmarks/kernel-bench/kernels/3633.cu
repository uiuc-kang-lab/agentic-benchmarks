#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-level reduction for minimum for float
__device__ inline float warp_reduce_min(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// Warp-level reduction for maximum for float
__device__ inline float warp_reduce_max(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// Warp-level reduction for minimum for double
__device__ inline double warp_reduce_min(double val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// Warp-level reduction for maximum for double
__device__ inline double warp_reduce_max(double val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// CUDA kernel applying HardSigmoid activation: y = clamp((x + 3) / 6, 0, 1).
// It uses warp-level primitives to detect if an entire warp's inputs are saturated.
// If all values in a warp are >= 3, then y = 1; if all <= -3, then y = 0.
// This avoids redundant per-thread arithmetic when the condition holds uniformly in the warp.

template <typename scalar_t>
__global__ void warp_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                          scalar_t* __restrict__ output,
                                          size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    // Determine lane id within the warp
    int lane = threadIdx.x & (warpSize - 1);
    
    // Compute warp-level min and max of the input values within the warp
    scalar_t warp_min = warp_reduce_min(x);
    scalar_t warp_max = warp_reduce_max(x);
    
    // Use a sentinel value (-1) which is outside the valid [0,1] output range
    // to decide if the entire warp falls in a saturated region.
    scalar_t warp_result = static_cast<scalar_t>(-1);
    if (lane == 0) {
      if (warp_min >= static_cast<scalar_t>(3)) {
        warp_result = static_cast<scalar_t>(1);
      } else if (warp_max <= static_cast<scalar_t>(-3)) {
        warp_result = static_cast<scalar_t>(0);
      }
    }
    // Broadcast the warp decision to all lanes
    warp_result = __shfl_sync(0xffffffff, warp_result, 0);
    
    scalar_t result;
    if (warp_result != static_cast<scalar_t>(-1)) {
      result = warp_result; // Uniform saturation in the warp
    } else {
      // Compute HardSigmoid normally: y = clamp((x+3)/6, 0, 1)
      result = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
      result = (result < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) :
               (result > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1) : result;
    }
    output[i] = result;
  }
}

// Host function that dispatches the kernel
torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  const size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "warp_hardsigmoid_cuda", ([&] {
    warp_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA) with warp-level optimization");
}
