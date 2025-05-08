#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Branchless absolute value function for device code
template <typename scalar_t>
__device__ inline scalar_t branchless_abs(scalar_t x) {
  if constexpr (std::is_same<scalar_t, float>::value) {
    return fabsf(x);
  } else {
    return fabs(x);
  }
}

// Branchless max with 0: max(x, 0) = (x + |x|) / 2
template <typename scalar_t>
__device__ inline scalar_t branchless_max_val(scalar_t x) {
  return (x + branchless_abs(x)) / static_cast<scalar_t>(2);
}

// Branchless min with 1: min(x, 1) = (x + 1 - |x-1|) / 2
template <typename scalar_t>
__device__ inline scalar_t branchless_min_val(scalar_t x) {
  return (x + static_cast<scalar_t>(1) - branchless_abs(x - static_cast<scalar_t>(1))) / static_cast<scalar_t>(2);
}

// Branchless HardSigmoid: y = clamp((x + 3)/6, 0, 1) computed without divergent branching
template <typename scalar_t>
__device__ inline scalar_t branchless_hardsigmoid_fn(scalar_t x) {
  scalar_t tmp = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
  // Apply branchless clamp: first clamp to >= 0, then clamp to <= 1
  return branchless_min_val(branchless_max_val(tmp));
}

// CUDA kernel applying HardSigmoid activation using branchless arithmetic operations
template <typename scalar_t>
__global__ void branchless_direct_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                                      scalar_t* __restrict__ output,
                                                      size_t numel) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    output[i] = branchless_hardsigmoid_fn(x);
  }
}

// Host function dispatching the kernel
torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "branchless_direct_hardsigmoid_cuda", ([&] {
    branchless_direct_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA) using branchless expressions");
}
