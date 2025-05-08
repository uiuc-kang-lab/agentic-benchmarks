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

  for (size_t i = idx; i < numel; i += stride * 4) {
    scalar_t x1 = input[i];
    scalar_t x2 = (i + stride < numel) ? input[i + stride] : 0;
    scalar_t x3 = (i + 2 * stride < numel) ? input[i + 2 * stride] : 0;
    scalar_t x4 = (i + 3 * stride < numel) ? input[i + 3 * stride] : 0;

    scalar_t y1 = fminf(fmaxf((x1 + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6), static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
    scalar_t y2 = (x2 + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
    scalar_t y3 = (x3 + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
    scalar_t y4 = (x4 + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);

    y1 = y1 < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) 
         : (y1 > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y1);
    y2 = y2 < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) 
         : (y2 > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y2);
    y3 = y3 < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) 
         : (y3 > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y3);
    y4 = y4 < static_cast<scalar_t>(0) ? static_cast<scalar_t>(0) 
         : (y4 > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y4);

    output[i] = y1;
    if (i + stride < numel) output[i + stride] = y2;
    if (i + 2 * stride < numel) output[i + 2 * stride] = y3;
    if (i + 3 * stride < numel) output[i + 3 * stride] = y4;
  }
}

torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  const size_t numel = input.numel();
  const int threads = 256;
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
