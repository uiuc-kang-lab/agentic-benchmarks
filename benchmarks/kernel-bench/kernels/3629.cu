#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Branchless clamp function using CUDA intrinsics to minimize warp divergence
template <typename scalar_t>
__device__ inline scalar_t clamp_val(scalar_t x) {
  if constexpr (std::is_same<scalar_t, float>::value) {
    return fminf(fmaxf(x, 0.f), 1.f);
  } else {
    return fmin(fmax(x, static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
  }
}

// CUDA kernel: computes HardSigmoid activation: y = clamp((x + 3) / 6, 0, 1) 
// using branchless intrinsics to reduce warp divergence
template <typename scalar_t>
__global__ void branchless_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               size_t numel) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    const scalar_t x = input[i];
    scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
    // Apply branchless clamp to maintain uniform control flow
    y = clamp_val(y);
    output[i] = y;
  }
}

// Forward function called from Python
torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  const size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "branchless_hardsigmoid_cuda", ([&] {
    branchless_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA) with branchless clamping");
}
