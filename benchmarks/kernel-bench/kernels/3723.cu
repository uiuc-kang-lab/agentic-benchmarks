#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with grid-stride loop for HardSigmoid activation
template <typename scalar_t>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     size_t numel) {
  // Compute global thread index and stride
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Grid-stride loop
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    // Compute (x + 3) / 6
    scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
    // Clamp the result between 0 and 1
    y = y < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0)
        : (y > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y);
    output[i] = y;
  }
}

// PyTorch forward function for HardSigmoid activation
torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  size_t numel = input.numel();
  
  // Launch parameters: 256 threads per block
  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", ([&] {
    hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
      input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA)");
}
