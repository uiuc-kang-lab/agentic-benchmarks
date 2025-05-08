#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ inline scalar_t warp_reduce_min(scalar_t val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    val = min(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template <typename scalar_t>
__device__ inline scalar_t warp_reduce_max(scalar_t val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template <typename scalar_t>
__global__ void optimized_hardsigmoid_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const size_t numel) {
    
  extern __shared__ char shared_mem[];
  scalar_t* s_data = reinterpret_cast<scalar_t*>(shared_mem);
  
  const size_t tid = threadIdx.x;
  const size_t idx = blockIdx.x * blockDim.x + tid;
  const size_t stride = blockDim.x * gridDim.x;
  const int lane = tid & (warpSize - 1);
  const int wid = tid / warpSize;
  
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    
    scalar_t warp_min = warp_reduce_min(x);
    scalar_t warp_max = warp_reduce_max(x);
    
    bool warp_saturated = false;
    scalar_t result;
    
    if (lane == 0) {
      if (warp_min >= static_cast<scalar_t>(3)) {
        result = static_cast<scalar_t>(1);
        warp_saturated = true;
      } else if (warp_max <= static_cast<scalar_t>(-3)) {
        result = static_cast<scalar_t>(0);
        warp_saturated = true;
      }
    }
    
    warp_saturated = __shfl_sync(0xffffffff, warp_saturated, 0);
    if (warp_saturated) {
      result = __shfl_sync(0xffffffff, result, 0);
    } else {
      result = (x + static_cast<scalar_t>(3)) * (static_cast<scalar_t>(1.0/6.0));
      result = max(static_cast<scalar_t>(0), min(static_cast<scalar_t>(1), result));
    }
    
    output[i] = result;
  }
}

torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  const size_t numel = input.numel();
  
  const int threads = 256;
  const int blocks = std::min(65535, (int)((numel + threads - 1) / threads));
  const size_t shared_mem_size = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_hardsigmoid_cuda", ([&] {
    optimized_hardsigmoid_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized HardSigmoid activation forward (CUDA)");
}