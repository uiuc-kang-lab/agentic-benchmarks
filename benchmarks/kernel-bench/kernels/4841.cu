#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Warp-level reduction function using shuffle intrinsics
__device__ inline float warpReduceSum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// CUDA kernel for L1 normalization with uniform control flow
// This kernel minimizes warp divergence by precomputing the main bound for vectorized processing
// and splitting the work into two loops (vectorized and remainder) that all threads execute uniformly.
__global__ void l1_norm_forward_kernel_uniform(const float* __restrict__ x,
                                                 float* __restrict__ out,
                                                 int N,
                                                 int D) {
  int row = blockIdx.x;  // Each block processes one row
  float thread_sum = 0.0f;

  // Compute the largest multiple of 4 that is <= D
  int main_bound = D & ~3;  // equivalent to D - (D % 4)

  // Vectorized loop: process the main part in groups of 4 using float4 loads if available
  if (main_bound > 0) {
    int vec_count = main_bound / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      thread_sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
    }
  }

  // Remainder loop: process remaining elements, if any
  for (int col = main_bound + threadIdx.x; col < D; col += blockDim.x) {
    float data = __ldg(&x[row * D + col]);
    thread_sum += fabsf(data);
  }

  // Perform warp-level reduction using shuffle intrinsics (all threads in warp execute uniformly)
  thread_sum = warpReduceSum(thread_sum);

  // Shared memory reduction across warps within the block
  extern __shared__ float sdata[];
  int warp_id = threadIdx.x / WARP_SIZE;
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
    sdata[warp_id] = thread_sum;
  }
  __syncthreads();

  // Final reduction by the first warp
  float total_sum = 0.0f;
  int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  if (threadIdx.x < num_warps) {
    total_sum = sdata[threadIdx.x];
  }
  total_sum = warpReduceSum(total_sum);

  // Thread 0 stores the final norm in shared memory ensuring uniform access
  if (threadIdx.x == 0) {
    sdata[0] = (total_sum == 0.0f ? 1e-12f : total_sum);
  }
  __syncthreads();
  total_sum = sdata[0];

  // Normalize the row elements
  // Use vectorized stores for the main part if available
  if (main_bound > 0) {
    int vec_count = main_bound / 4;
    float4* out_vec = reinterpret_cast<float4*>(out);
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      data.x /= total_sum;
      data.y /= total_sum;
      data.z /= total_sum;
      data.w /= total_sum;
      out_vec[row * vec_count + i] = data;
    }
  }
  // Remainder normalization
  for (int col = main_bound + threadIdx.x; col < D; col += blockDim.x) {
    float val = __ldg(&x[row * D + col]);
    out[row * D + col] = val / total_sum;
  }
}

// Host function to launch the kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Determine threads per block.
  // If D is divisible by 4, we can use min(D/4, 1024) for the vectorized loop;
  // otherwise, we use min(D, 1024) for the scalar loop. This decision is uniform across the block.
  int threads = (D < 1024) ? D : 1024;
  if ((D & 3) == 0) {
    int vec_count = D / 4;
    if (vec_count < threads) {
      threads = vec_count;
    }
  }

  // Calculate shared memory size (one float per warp)
  int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
  int shared_mem_size = num_warps * sizeof(float);

  // Launch one block per row
  l1_norm_forward_kernel_uniform<<<N, threads, shared_mem_size>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "L1 Normalization forward pass with uniform control flow (CUDA)");
}
