#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Inline clamp function that works for both float and double
template <typename scalar_t>
__device__ inline scalar_t my_clamp(scalar_t x) {
  if constexpr (std::is_same<scalar_t, float>::value) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
  } else {
    return fmin(fmax(x, static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
  }
}

// CUDA kernel that leverages shared memory to load a tile of input data
// and then applies the HardSigmoid activation function: y = clamp((x + 3)/6, 0, 1)
// Each block cooperatively loads data into shared memory to reduce global memory latency.

template <typename scalar_t>
__global__ void shared_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                            scalar_t* __restrict__ output,
                                            size_t numel) {
  // Dynamically allocated shared memory tile
  extern __shared__ scalar_t tile[];
  int tid = threadIdx.x;

  // Grid-stride loop: each block processes multiple consecutive tiles
  for (size_t base = blockIdx.x * blockDim.x; base < numel; base += gridDim.x * blockDim.x) {
    size_t idx = base + tid;
    // Load a tile of input from global memory into shared memory
    if (idx < numel) {
      tile[tid] = input[idx];
    }
    __syncthreads();

    // Process the loaded tile: apply HardSigmoid activation
    if (idx < numel) {
      scalar_t x = tile[tid];
      scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
      y = my_clamp<scalar_t>(y);
      output[idx] = y;
    }
    __syncthreads();
  }
}

// Host function to launch the kernel

torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  size_t numel = input.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "shared_hardsigmoid_cuda", ([&] {
    shared_hardsigmoid_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA) using shared memory");
}
