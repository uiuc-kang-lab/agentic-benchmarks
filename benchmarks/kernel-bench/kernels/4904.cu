#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

__global__ void l1_norm_forward_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int N,
                                       int D) {
  int row = blockIdx.x;
  float sum = 0.0f;

  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    float val = x[row * D + col];
    sum += fabsf(val);
  }

  sum = warpReduceSum(sum);

  if (threadIdx.x % warpSize == 0) {
    atomicAdd(out + row, sum);
  }
  __syncthreads();

  float total_sum = out[row];
  __syncthreads();

  if (threadIdx.x < D) {
    val = x[row * D + threadIdx.x];
    out[row * D + threadIdx.x] = val / total_sum;
  }
}

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
  x = x.contiguous();

  auto out = torch::zeros_like(x);
  int N = x.size(0);
  int D = x.size(1);

  int threads = std::min<int>(1024, D);

  l1_norm_forward_kernel<<<N, threads>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "L1 Normalization forward pass (CUDA)");
}
