#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// CUDA kernel implementing L1 normalization using warp shuffle reductions
__global__ void l1_norm_forward_kernel(const float* __restrict__ x,
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

  // Allocate shared memory to store partial sums from each warp
  extern __shared__ float sdata[];
  int warpId = threadIdx.x / WARP_SIZE;
  if (lane == 0) {
    sdata[warpId] = thread_sum;
  }
  __syncthreads();

  // Thread 0 aggregates the results from all warps
  if (threadIdx.x == 0) {
    float total_sum = 0.0f;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    for (int i = 0; i < num_warps; i++) {
      total_sum += sdata[i];
    }
    // Avoid division by zero by adjusting total_sum
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
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Determine the number of threads per block
  int threads = std::min<int>(1024, D);
  int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
  int shared_mem_size = num_warps * sizeof(float);

  l1_norm_forward_kernel<<<N, threads, shared_mem_size>>>(
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
