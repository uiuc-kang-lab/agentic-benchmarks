#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Vectorized and coalesced HardSigmoid kernel
// Computes y = clamp((x + 3) / 6, 0, 1) using vectorized global memory accesses

template <typename scalar_t>
__global__ void vectorized_coalesced_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                                           scalar_t* __restrict__ output,
                                                           size_t numel) {
  // Calculate global thread index and stride
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Use vectorized loads/stores for memory coalescing based on precision
  if constexpr (std::is_same<scalar_t, float>::value) {
    // For float, use float4 (4 floats at a time)
    constexpr int vecSize = 4;
    using vec_t = float4;
    size_t num_vec = numel / vecSize;  // number of vectorized elements

    // Process main vectorized portion
    for (size_t i = idx; i < num_vec; i += stride) {
      vec_t in_vec = reinterpret_cast<const vec_t*>(input)[i];
      vec_t out_vec;
      out_vec.x = fminf(fmaxf((in_vec.x + 3.0f) / 6.0f, 0.0f), 1.0f);
      out_vec.y = fminf(fmaxf((in_vec.y + 3.0f) / 6.0f, 0.0f), 1.0f);
      out_vec.z = fminf(fmaxf((in_vec.z + 3.0f) / 6.0f, 0.0f), 1.0f);
      out_vec.w = fminf(fmaxf((in_vec.w + 3.0f) / 6.0f, 0.0f), 1.0f);
      reinterpret_cast<vec_t*>(output)[i] = out_vec;
    }

    // Process any remaining elements
    size_t tail_start = num_vec * vecSize;
    for (size_t i = idx; i < (numel - tail_start); i += stride) {
      size_t index = tail_start + i;
      float x = input[index];
      float y = (x + 3.0f) / 6.0f;
      y = fminf(fmaxf(y, 0.0f), 1.0f);
      output[index] = y;
    }
  } else {
    // For double, use double2 (2 doubles at a time)
    constexpr int vecSize = 2;
    using vec_t = double2;
    size_t num_vec = numel / vecSize;

    for (size_t i = idx; i < num_vec; i += stride) {
      vec_t in_vec = reinterpret_cast<const vec_t*>(input)[i];
      vec_t out_vec;
      out_vec.x = fmin(fmax((in_vec.x + 3.0) / 6.0, 0.0), 1.0);
      out_vec.y = fmin(fmax((in_vec.y + 3.0) / 6.0, 0.0), 1.0);
      reinterpret_cast<vec_t*>(output)[i] = out_vec;
    }

    // Handle tail elements for double
    size_t tail_start = num_vec * vecSize;
    for (size_t i = idx; i < (numel - tail_start); i += stride) {
      size_t index = tail_start + i;
      double x = input[index];
      double y = (x + 3.0) / 6.0;
      y = fmin(fmax(y, 0.0), 1.0);
      output[index] = y;
    }
  }
}

// Host function to launch the vectorized kernel

torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "vectorized_coalesced_hardsigmoid_cuda", ([&] {
    vectorized_coalesced_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      numel
    );
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Vectorized and coalesced HardSigmoid activation (CUDA)");
}
