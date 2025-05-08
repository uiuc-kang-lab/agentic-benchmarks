#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

// Store frequently accessed, read-only constants in constant memory
__constant__ float TanhThreshold_f = 8.f;
__constant__ float One_f = 1.f;
__constant__ float NegOne_f = -1.f;

__constant__ double TanhThreshold_d = 8.0;
__constant__ double One_d = 1.0;
__constant__ double NegOne_d = -1.0;

// Helper device function to apply tanh with saturation using constant memory values
template <typename scalar_t>
__device__ __forceinline__ scalar_t apply_tanh(scalar_t x) {
  if constexpr (std::is_same<scalar_t, float>::value) {
    // Use constants from constant memory for float
    if (x > TanhThreshold_f) return One_f;
    else if (x < NegOne_f) return NegOne_f;
    else return tanhf(x);
  } else {
    // Use constants from constant memory for double
    if (x > TanhThreshold_d) return One_d;
    else if (x < NegOne_d) return NegOne_d;
    else return tanh(x);
  }
}

// CUDA kernel that uses vectorized memory accesses and constant memory constants
// For float types, we use float4 vectorization; for double, we use double2 vectorization.

template <typename scalar_t>
__global__ void tanh_kernel_const_vectorized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if constexpr (std::is_same<scalar_t, float>::value) {
    // Process 4 floats at a time using float4
    int vecSize = size / 4;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    for (int i = idx; i < vecSize; i += stride) {
      float4 in_val = input4[i];
      float4 out_val;
      out_val.x = apply_tanh<float>(in_val.x);
      out_val.y = apply_tanh<float>(in_val.y);
      out_val.z = apply_tanh<float>(in_val.z);
      out_val.w = apply_tanh<float>(in_val.w);
      output4[i] = out_val;
    }

    // Handle remaining elements
    int remaining = vecSize * 4;
    for (int i = remaining + idx; i < size; i += stride) {
      output[i] = apply_tanh<float>(input[i]);
    }
  } else {
    // For double, use double2 vectorization
    int vecSize = size / 2;
    const double2* input2 = reinterpret_cast<const double2*>(input);
    double2* output2 = reinterpret_cast<double2*>(output);
    for (int i = idx; i < vecSize; i += stride) {
      double2 in_val = input2[i];
      double2 out_val;
      out_val.x = apply_tanh<double>(in_val.x);
      out_val.y = apply_tanh<double>(in_val.y);
      output2[i] = out_val;
    }
    // Process remaining element if the total size is odd
    int remaining = vecSize * 2;
    for (int i = remaining + idx; i < size; i += stride) {
      output[i] = apply_tanh<double>(input[i]);
    }
  }
}

// Forward function wrapping the kernel launch

torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int threads = 256;
  int size = input.numel();
  int blocks;

  // Determine grid size based on vectorization factor
  if (input.scalar_type() == torch::kFloat32) {
    blocks = ((size / 4) + threads - 1) / threads;
  } else {
    blocks = ((size / 2) + threads - 1) / threads;
  }

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_const_vectorized", ([&] {
    tanh_kernel_const_vectorized<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        size
    );
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Tanh forward with vectorization and constant memory optimization (CUDA)");
}
