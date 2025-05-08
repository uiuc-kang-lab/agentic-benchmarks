#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Branchless clamp function using CUDA intrinsics
template <typename scalar_t>
__device__ inline scalar_t clamp_val(scalar_t x) {
  if constexpr (std::is_same<scalar_t, float>::value) {
    return fminf(fmaxf(x, 0.f), 1.f);
  } else {
    return fmin(fmax(x, static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
  }
}

// CUDA kernel combining branchless clamping with loop unrolling (factor 4) for improved performance.
template <typename scalar_t>
__global__ void unrolled_branchless_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                                        scalar_t* __restrict__ output,
                                                        size_t numel) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  size_t i = idx;

  // Unroll loop by factor 4 to reduce loop overhead and increase ILP
  for (; i + 3 * stride < numel; i += 4 * stride) {
    scalar_t x0 = input[i];
    scalar_t x1 = input[i + stride];
    scalar_t x2 = input[i + 2 * stride];
    scalar_t x3 = input[i + 3 * stride];

    scalar_t y0 = clamp_val((x0 + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6));
    scalar_t y1 = clamp_val((x1 + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6));
    scalar_t y2 = clamp_val((x2 + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6));
    scalar_t y3 = clamp_val((x3 + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6));

    output[i]             = y0;
    output[i + stride]    = y1;
    output[i + 2 * stride]= y2;
    output[i + 3 * stride]= y3;
  }

  // Process remaining elements
  for (; i < numel; i += stride) {
    scalar_t x = input[i];
    scalar_t y = clamp_val((x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6));
    output[i] = y;
  }
}

// Forward function exposed to Python
torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "unrolled_branchless_hardsigmoid_cuda", ([&] {
    unrolled_branchless_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA) with branchless clamping and loop unrolling");
}
