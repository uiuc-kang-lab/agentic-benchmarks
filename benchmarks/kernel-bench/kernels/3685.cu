#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Store frequently accessed constants in constant memory
__constant__ float c_three_f = 3.0f;
__constant__ float c_inv6_f = 0.16666667f;
__constant__ double c_three_d = 3.0;
__constant__ double c_inv6_d = 0.16666666666666666;

// Template specialization for fetching the constant value of 3
template <typename scalar_t>
__device__ __forceinline__ scalar_t get_three();

template <>
__device__ __forceinline__ float get_three<float>() {
  return c_three_f;
}

template <>
__device__ __forceinline__ double get_three<double>() {
  return c_three_d;
}

// Template specialization for fetching the constant value of 1/6
template <typename scalar_t>
__device__ __forceinline__ scalar_t get_inv6();

template <>
__device__ __forceinline__ float get_inv6<float>() {
  return c_inv6_f;
}

template <>
__device__ __forceinline__ double get_inv6<double>() {
  return c_inv6_d;
}

// CUDA kernel: computes HardSigmoid activation: y = clamp((x + 3) / 6, 0, 1)
// Using constants stored in __constant__ memory
template <typename scalar_t>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                    scalar_t* __restrict__ output,
                                    size_t numel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    scalar_t three = get_three<scalar_t>();
    scalar_t inv6 = get_inv6<scalar_t>();
    scalar_t y = (x + three) * inv6;
    y = (y < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) 
         : ((y > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1) : y);
    output[i] = y;
  }
}

// Host function to launch the CUDA kernel
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
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA) with constant memory");
}
