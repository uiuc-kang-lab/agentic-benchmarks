// This CUDA code combines branchless clamping with optional vectorized memory accesses for float tensors.
// For float type with a size that's a multiple of 4, we use float4 loads/stores to improve memory throughput,
// while for other types or non-multiple-of-4 cases, we fallback to a scalar kernel with branchless control flow.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Branchless clamp function inlined for efficiency
template <typename scalar_t>
__forceinline__ __device__ scalar_t branchless_clamp(scalar_t x) {
  if constexpr (std::is_same<scalar_t, float>::value) {
    return fminf(fmaxf(x, 0.f), 1.f);
  } else {
    return fmin(fmax(x, static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
  }
}

// Optimized kernel for float type using vectorized float4 accesses
__global__ void optimized_hardsigmoid_kernel_float(const float* __restrict__ input,
                                                     float* __restrict__ output,
                                                     size_t numel) {
  // Number of float4 elements
  size_t vec_size = numel / 4;
  const float4* input_vec = reinterpret_cast<const float4*>(input);
  float4* output_vec = reinterpret_cast<float4*>(output);

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Process vectorized portion
  for (size_t i = idx; i < vec_size; i += stride) {
    float4 in_val = input_vec[i];
    float4 out_val;
    out_val.x = fminf(fmaxf((in_val.x + 3.f) / 6.f, 0.f), 1.f);
    out_val.y = fminf(fmaxf((in_val.y + 3.f) / 6.f, 0.f), 1.f);
    out_val.z = fminf(fmaxf((in_val.z + 3.f) / 6.f, 0.f), 1.f);
    out_val.w = fminf(fmaxf((in_val.w + 3.f) / 6.f, 0.f), 1.f);
    output_vec[i] = out_val;
  }

  // Tail processing for remaining elements if numel is not divisible by 4
  size_t offset = vec_size * 4;
  for (size_t i = idx; i < (numel - offset); i += stride) {
    float x = input[offset + i];
    float y = (x + 3.f) / 6.f;
    output[offset + i] = fminf(fmaxf(y, 0.f), 1.f);
  }
}

// Generic kernel for non-float types or when vectorization is not applicable
template <typename scalar_t>
__global__ void optimized_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
    y = branchless_clamp(y);
    output[i] = y;
  }
}

// C++ interface (forward function) exposed to Python
torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  const size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  // Use vectorized kernel if type is float and the number of elements is a multiple of 4
  if (input.scalar_type() == at::ScalarType::Float && (numel % 4 == 0)) {
    optimized_hardsigmoid_kernel_float<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);
  } else {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_hardsigmoid_cuda", ([&] {
      optimized_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          numel);
    }));
  }

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized HardSigmoid activation forward (CUDA)");
}
