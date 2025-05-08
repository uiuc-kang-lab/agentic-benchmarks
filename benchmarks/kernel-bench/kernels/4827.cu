#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Optimized kernel using warp-level reduction and tunable block size
__global__ void l1_norm_forward_kernel_optimized(const float* __restrict__ x,
                                                   float* __restrict__ out,
                                                   int N,
                                                   int D) {
  // Each block processes one row
  int row = blockIdx.x;
  
  // Declare shared memory for warp-level sums; one float per warp
  extern __shared__ float sdata[];
  
  // Each thread computes a partial sum of absolute values
  float sum_val = 0.0f;
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    float val = x[row * D + col];
    sum_val += fabsf(val);
  }

  // Warp-level reduction using shuffle intrinsics
  unsigned int lane = threadIdx.x & 31; // lane index in the warp
  for (int offset = 16; offset > 0; offset /= 2) {
    sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
  }

  // Write reduced sum of each warp to shared memory
  int warp_id = threadIdx.x / 32;
  if (lane == 0) {
    sdata[warp_id] = sum_val;
  }
  __syncthreads();

  // Final reduction: let first warp combine the per-warp sums
  float total_sum = 0.0f;
  int num_warps = (blockDim.x + 31) / 32;
  if (threadIdx.x < num_warps) {
    total_sum = sdata[threadIdx.x];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = 1; i < num_warps; ++i) {
      total_sum += sdata[i];
    }
    // Avoid division by zero by setting a small epsilon
    if (total_sum == 0.0f) {
      total_sum = 1e-12f;
    }
    sdata[0] = total_sum;
  }
  __syncthreads();
  total_sum = sdata[0];

  // Normalize the row using the computed L1 norm
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    out[row * D + col] = x[row * D + col] / total_sum;
  }
}

// Host function: selects block size from candidate options and launches the optimized kernel

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this kernel.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Experimented block sizes for optimal performance on NVIDIA H100
  int block_candidates[5] = {32, 64, 128, 256, 512};
  int block_size = 32; // default
  for (int i = 0; i < 5; ++i) {
    if (D >= block_candidates[i]) {
      block_size = block_candidates[i];
    } else {
      break;
    }
  }

  // Shared memory: one float per warp
  int num_warps = (block_size + 31) / 32;
  int shared_mem_size = num_warps * sizeof(float);

  // Launch one block per row
  l1_norm_forward_kernel_optimized<<<N, block_size, shared_mem_size>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "L1 Normalization forward pass optimized (CUDA)");
}
