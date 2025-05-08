#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Store frequently used constants in constant memory for low-latency access
__constant__ float const_params_f[2] = {3.0f, 6.0f};
__constant__ double const_params_d[2] = {3.0, 6.0};

// CUDA kernel: computes HardSigmoid activation using constants from __constant__ memory
template <typename scalar_t>
__global__ void hardsigmoid_kernel_const(const scalar_t* __restrict__ input,
                                           scalar_t* __restrict__ output,
                                           size_t numel) {
  // Load constants from __constant__ memory based on data type
  scalar_t offset, scale;
  if constexpr (std::is_same<scalar_t, float>::value) {
    offset = const_params_f[0];
    scale  = const_params_f[1];
  } else if constexpr (std::is_same<scalar_t, double>::value) {
    offset = const_params_d[0];
    scale  = const_params_d[1];
  }

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    scalar_t y = (x + offset) / scale;
    y = y < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) : (y > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y);
    output[i] = y;
  }
}

// PyTorch forward function
torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  const size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda_const", ([&] {
    hardsigmoid_kernel_const<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA) using constant memory");
}
