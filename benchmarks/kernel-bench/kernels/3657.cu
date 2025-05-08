/*
 * This CUDA kernel implements the HardSigmoid activation fused with warp-level predicate checks.
 * It combines the simple strided-loop design of Kernel 1 with an optimized saturation check inspired by Kernel 2.
 * The warp-level check uses __all_sync to determine if all threads in the warp have inputs in the saturated range,
 * allowing the kernel to bypass expensive per-thread arithmetic when possible.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized hard sigmoid kernel using warp-level predicate shortcuts
// y = clamp((x + 3) / 6, 0, 1)

template <typename scalar_t>
__global__ void fast_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                          scalar_t* __restrict__ output,
                                          size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Process elements in a strided loop
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];

    // Use warp-level predicate checks instead of explicit warp reductions
    // Here, each thread in the warp processes its own element. The __all_sync
    // function checks if all active threads in the warp satisfy the saturation condition.
    bool warp_all_high = __all_sync(0xffffffff, (x >= static_cast<scalar_t>(3)));
    bool warp_all_low  = __all_sync(0xffffffff, (x <= static_cast<scalar_t>(-3)));

    scalar_t y;
    if (warp_all_high) {
      // All threads in this warp have x >= 3 so output saturates to 1
      y = static_cast<scalar_t>(1);
    } else if (warp_all_low) {
      // All threads in this warp have x <= -3 so output saturates to 0
      y = static_cast<scalar_t>(0);
    } else {
      // Otherwise, compute the HardSigmoid normally
      y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
      y = (y < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) :
          (y > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1) : y;
    }
    output[i] = y;
  }
}

// Host launch function

torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  const size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fast_hardsigmoid_cuda", ([&] {
    fast_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA) optimized with warp-level predicate");
}
