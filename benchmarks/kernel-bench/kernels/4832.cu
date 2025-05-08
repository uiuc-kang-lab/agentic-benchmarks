#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// CUDA kernel implementing L1 normalization with stride loops
__global__ void l1_norm_forward_kernel_stride(const float* __restrict__ x,
                                               float* __restrict__ out,
                                               int N,
                                               int D) {
  int row = blockIdx.x;
  float thread_sum = 0.0f;

  // Use stride loop to cover all columns even when D > blockDim.x
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    float val = x[row * D + col];
    thread_sum += fabsf(val);
  }

  // Warp-level reduction using shuffle intrinsics
  unsigned int lane = threadIdx.x % WARP_SIZE;
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
  }

  // Shared memory reduction across warps
  extern __shared__ float sdata[];
  int warp_id = threadIdx.x / WARP_SIZE;
  if (lane == 0) {
    sdata[warp_id] = thread_sum;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float total_sum = 0.0f;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    for (int i = 0; i < num_warps; i++) {
      total_sum += sdata[i];
    }
    if (total_sum == 0.0f) {
      total_sum = 1e-12f;
    }
    sdata[0] = total_sum;
  }
  __syncthreads();
  float total_sum = sdata[0];

  // Normalize the row elements using a stride loop to handle boundary conditions
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    out[row * D + col] = x[row * D + col] / total_sum;
  }
}

// Host function to launch the CUDA kernel

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Determine the number of threads per block (up to 1024, but not more than D)
  int threads = (D < 1024) ? D : 1024;
  int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
  int shared_mem_size = num_warps * sizeof(float);

  l1_norm_forward_kernel_stride<<<N, threads, shared_mem_size>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "L1 Normalization forward pass with stride loops (CUDA)");
}
