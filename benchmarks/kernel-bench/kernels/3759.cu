#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Store threshold constants in constant memory
__constant__ float c_upper_threshold_float = 20.0f;
__constant__ float c_lower_threshold_float = -20.0f;
__constant__ double c_upper_threshold_double = 20.0;
__constant__ double c_lower_threshold_double = -20.0;

// Define optimal thread block size for modern GPUs (e.g., H100) and unroll factor
const int OPTIMAL_THREADS = 512;
const int UNROLL_FACTOR = 4;

// Combined CUDA kernel using loop unrolling and constant memory thresholds
template <typename scalar_t>
__global__ void softplus_kernel_combined(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Loop unrolling by a factor of UNROLL_FACTOR
  for (; idx < size; idx += stride * UNROLL_FACTOR) {
    #pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
      int index = idx + i * stride;
      if (index < size) {
        scalar_t x = input[index];
        if constexpr (std::is_same<scalar_t, float>::value) {
          if (x > c_upper_threshold_float) {
            output[index] = x;
          } else if (x < c_lower_threshold_float) {
            output[index] = expf(x);
          } else {
            output[index] = log1pf(expf(x));
          }
        } else {
          if (x > c_upper_threshold_double) {
            output[index] = x;
          } else if (x < c_lower_threshold_double) {
            output[index] = exp(x);
          } else {
            output[index] = log1p(exp(x));
          }
        }
      }
    }
  }
}

// CUDA forward function
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int size = input.numel();
  // Calculate number of blocks factoring in both optimal threads and unroll factor
  const int blocks = (size + OPTIMAL_THREADS * UNROLL_FACTOR - 1) / (OPTIMAL_THREADS * UNROLL_FACTOR);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
    softplus_kernel_combined<scalar_t><<<blocks, OPTIMAL_THREADS>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        size);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
