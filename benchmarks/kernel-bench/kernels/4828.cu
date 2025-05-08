#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

__global__ void l1_norm_forward_kernel_optimized(const float* __restrict__ x,
                                                 float* __restrict__ out,
                                                 int N,
                                                 int D) {
  int row = blockIdx.x;
  float thread_sum = 0.0f;

  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    float val = x[row * D + col];
    thread_sum += fabsf(val);
  }

  // Perform warp-level reduction
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
  }

  // Allocate shared memory for partial warp sums
  extern __shared__ float sdata[];
  int warpId = threadIdx.x / WARP_SIZE;
  if (threadIdx.x % WARP_SIZE == 0) {
    sdata[warpId] = thread_sum;
  }

  // Sync before full block reduction only if multiple warps
  if (blockDim.x / WARP_SIZE > 1) {
    __syncthreads();
  }

  // Final reduction across warps
  if (threadIdx.x < WARP_SIZE) {
    float total_sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? sdata[threadIdx.x] : 0.0f;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      total_sum += __shfl_down_sync(0xffffffff, total_sum, offset);
    }

    // Write result to shared memory only once to minimize sync
    if (threadIdx.x == 0) {
      sdata[0] = total_sum == 0.0f ? 1e-12f : total_sum;
    }
  }

  __syncthreads();  // Use sync only once before row normalization

  float total_sum = sdata[0];
  
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    out[row * D + col] = x[row * D + col] / total_sum;
  }
}

// Host function to launch the optimized CUDA kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  int threads = std::min<int>(1024, D);
  int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
  int shared_mem_size = num_warps * sizeof(float);

  l1_norm_forward_kernel_optimized<<<N, threads, shared_mem_size>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "L1 Normalization forward pass with minimized sync (CUDA)");
}
