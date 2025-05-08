#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Device function for clamping value to [0, 1] for both float and double types
template <typename scalar_t>
__device__ inline scalar_t clamp_val(scalar_t x) {
  if constexpr (std::is_same<scalar_t, float>::value) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
  } else {
    return fmin(fmax(x, static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
  }
}

// CUDA kernel using shared memory to store frequently used constants
// Computes HardSigmoid: y = clamp((x + 3) / 6, 0, 1)
template <typename scalar_t>
__global__ void shared_mem_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               size_t numel) {
  // Declare shared memory for constants: [0] = 3, [1] = 1/6
  __shared__ scalar_t s_consts[2];
  if (threadIdx.x == 0) {
    s_consts[0] = static_cast<scalar_t>(3);
    s_consts[1] = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);
  }
  __syncthreads();

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    scalar_t y = (x + s_consts[0]) * s_consts[1];
    output[i] = clamp_val(y);
  }
}

// Host function launching the kernel
torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "shared_mem_hardsigmoid_cuda", ([&] {
    shared_mem_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Shared memory based HardSigmoid activation forward (CUDA)");
}
