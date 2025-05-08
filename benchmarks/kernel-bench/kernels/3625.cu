#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel: computes HardSigmoid activation: y = clamp((x + 3) / 6, 0, 1)
template <typename scalar_t>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
    y = y < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) 
         : (y > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y);
    output[i] = y;
  }
}

torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  const size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", ([&] {
    hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA)");
}