#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Device function using compile-time type selection to minimize divergence
// and perform branchless clamping
template <typename scalar_t>
__device__ inline scalar_t clamp_val(scalar_t x) {
  if constexpr (std::is_same<scalar_t, float>::value) {
    return fminf(fmaxf(x, 0.f), 1.f);
  } else {
    return fmin(fmax(x, static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
  }
}

// Kernel that applies HardSigmoid: y = clamp((x + 3) / 6, 0, 1)
// Uses atomic operations only when absolutely necessary
// This is a single-threaded operation, so atomic operations are not required

template <typename scalar_t>
__global__ void atomic_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                           scalar_t* __restrict__ output,
                                           size_t numel) {
  // Calculate global thread index and stride
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  
  // Precompute constants as constexpr to avoid redundant computations
  constexpr scalar_t add_const = static_cast<scalar_t>(3);
  constexpr scalar_t div_const = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);

  // Process elements in a stride loop for better load balancing among warps
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    scalar_t y = (x + add_const) * div_const;
    // Branchless clamping using our inline helper
    y = clamp_val(y);
    output[i] = y;  // No atomic operation needed here
  }
}

// Host function to dispatch the kernel
torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "atomic_hardsigmoid_cuda", ([&] {
    atomic_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Atomic HardSigmoid activation forward (CUDA)");
}