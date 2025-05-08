#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Device function using compile-time type selection to minimize divergence
// and perform branchless clamping
template <typename scalar_t>
__device__ inline scalar_t clamp_val(scalar_t x) {
  if constexpr (std::is_same<scalar_t, float>::value) {
    return fminf(fmaxf(x, 0.f), 1.f);
  } else {
    return fmin(fmax(x, static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
  }
}

// Kernel using shared memory to store constants
// Computes y = clamp((x + 3) / 6, 0, 1) using branchless operations
// Utilizes shared memory to reduce global memory latency

template <typename scalar_t>
__global__ void shared_memory_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                                 scalar_t* __restrict__ output,
                                                 size_t numel) {
  // Shared memory for constants
  __shared__ scalar_t shared_add_const;
  __shared__ scalar_t shared_div_const;

  // Initialize shared memory once per block
  if (threadIdx.x == 0) {
    shared_add_const = static_cast<scalar_t>(3);
    shared_div_const = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);
  }
  __syncthreads();

  // Calculate global thread index and stride
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  // Process elements in a stride loop for better load balancing among warps
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    scalar_t y = (x + shared_add_const) * shared_div_const;
    // Branchless clamping using our inline helper
    y = clamp_val(y);
    output[i] = y;
  }
}

// Host function to dispatch the kernel

torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "shared_memory_hardsigmoid_cuda", ([&] {
    shared_memory_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Shared Memory HardSigmoid activation forward (CUDA)");
}
