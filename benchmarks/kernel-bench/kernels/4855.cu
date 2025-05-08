#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Warp-level reduction using shuffle intrinsics
__device__ inline float warpReduceSum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Block-level reduction using shared memory
__device__ float blockReduceSum(float val) {
  __shared__ float shared[WARP_SIZE];  // Enough for 1024 threads (32 warps)
  int lane = threadIdx.x % WARP_SIZE;
  int wid  = threadIdx.x / WARP_SIZE;
  
  // Each warp reduces its own values
  val = warpReduceSum(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  // Let the first warp reduce the partial sums
  int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  val = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
  if (threadIdx.x < WARP_SIZE) {
    val = warpReduceSum(val);
  }
  __syncthreads();

  // Thread 0 finalizes reduction and avoids division by zero
  if (threadIdx.x == 0) {
    if (val == 0.0f) {
      val = 1e-12f;
    }
    shared[0] = val;
  }
  __syncthreads();
  return shared[0];
}

// Templated device function for computing partial L1 norm for a row
template<bool UseVectorized>
__device__ float computePartialSum(const float* __restrict__ x, int row, int D) {
  float sum = 0.0f;
  if constexpr (UseVectorized) {
    int vec_count = D / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
    }
  } else {
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      sum += fabsf(__ldg(&x[row * D + col]));
    }
  }
  return sum;
}

// Templated device function for normalizing a row with the computed L1 norm
template<bool UseVectorized>
__device__ void normalizeRow(const float* __restrict__ x,
                               float* __restrict__ out,
                               int row,
                               int D,
                               float norm) {
  if constexpr (UseVectorized) {
    int vec_count = D / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* out_vec = reinterpret_cast<float4*>(out);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      data.x /= norm;
      data.y /= norm;
      data.z /= norm;
      data.w /= norm;
      out_vec[row * vec_count + i] = data;
    }
  } else {
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      float val = __ldg(&x[row * D + col]);
      out[row * D + col] = val / norm;
    }
  }
}

// Global kernel launching one block per row, templated for vectorized vs scalar path
template<bool UseVectorized>
__global__ void l1_norm_forward_kernel_combined(const float* __restrict__ x,
                                                  float* __restrict__ out,
                                                  int N,
                                                  int D) {
  int row = blockIdx.x;
  if (row < N) {
    // Compute the partial L1 norm sum using either vectorized or scalar loads
    float partial = computePartialSum<UseVectorized>(x, row, D);
    // Reduce partial sums across the block
    float total = blockReduceSum(partial);
    // Normalize the row using the computed total L1 norm
    normalizeRow<UseVectorized>(x, out, row, D, total);
  }
}

// Host function to launch the combined kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
  x = x.contiguous();
  
  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);
  
  // Decide threads based on D (max 1024 threads per block)
  int threads = (D < 1024) ? D : 1024;
  bool use_vec = (D % 4 == 0);
  
  if (use_vec) {
    l1_norm_forward_kernel_combined<true><<<N, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), N, D);
  } else {
    l1_norm_forward_kernel_combined<false><<<N, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), N, D);
  }
  
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized L1 Normalization forward pass with templated vectorized loads (CUDA)");
}
