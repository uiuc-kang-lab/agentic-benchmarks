#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Kernel that computes L1 norm per row with coalesced global memory accesses
__global__ void coalesced_l1_norm_kernel(const float* __restrict__ x,
                                           float* __restrict__ out,
                                           int N,
                                           int D) {
  // Each block processes one row
  int row = blockIdx.x;
  if (row >= N) return;

  int base = row * D;
  float threadSum = 0.0f;

  // Coalesced global memory read: threads in a warp read consecutive elements
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    threadSum += fabsf(x[base + col]);
  }

  // Warp-level reduction using shuffle intrinsics
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
  }

  // Shared memory for block-level reduction
  extern __shared__ float sdata[];
  int warpId = threadIdx.x / WARP_SIZE;
  if ((threadIdx.x % WARP_SIZE) == 0) {
    sdata[warpId] = threadSum;
  }
  __syncthreads();

  float normSum = 0.0f;
  int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  if (threadIdx.x < numWarps) {
    float val = sdata[threadIdx.x];
    // Reduce the warp sums
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (threadIdx.x == 0) {
      normSum = (val == 0.0f ? 1e-12f : val);
      sdata[0] = normSum;
    }
  }
  __syncthreads();
  normSum = sdata[0];

  // Coalesced write: threads write consecutive elements
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    out[base + col] = x[base + col] / normSum;
  }
}

// Host function to launch the kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Choose number of threads: use at most 1024 threads per block
  int threads = (D < 1024) ? D : 1024;
  int numWarps = (threads + WARP_SIZE - 1) / WARP_SIZE;
  int shared_mem_size = numWarps * sizeof(float);

  // Launch one block per row
  coalesced_l1_norm_kernel<<<N, threads, shared_mem_size>>>(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      N,
      D);

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Coalesced L1 Normalization kernel (CUDA)");
}
