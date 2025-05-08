#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Warp-level reduction using shuffle intrinsics for efficient summation
__device__ inline float warpReduceSum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// CUDA kernel for L1 normalization that optimizes global memory loads/stores
// using __ldg() for read-only accesses and aligns accesses to 128-bit boundaries via float4.
__global__ void l1_norm_aligned_kernel(const float* __restrict__ x,
                                         float* __restrict__ out,
                                         int N,
                                         int D) {
  int row = blockIdx.x;
  extern __shared__ float sdata[];

  // Determine if we can use vectorized (128-bit aligned) loads/stores
  bool vec_possible = (D % 4 == 0);
  float sum_val = 0.0f;

  if (vec_possible) {
    int vec_count = D / 4;  // 4 floats = 16 bytes, 128-bit
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      // Use __ldg() for read-only load from global memory
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      sum_val += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
    }
  } else {
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      float data = __ldg(&x[row * D + i]);
      sum_val += fabsf(data);
    }
  }

  // In-warp reduction
  sum_val = warpReduceSum(sum_val);
  int lane = threadIdx.x % WARP_SIZE;
  int warpId = threadIdx.x / WARP_SIZE;
  if (lane == 0) {
    sdata[warpId] = sum_val;
  }
  __syncthreads();

  // Final reduction using the first warp
  if (threadIdx.x < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
    sum_val = sdata[lane];
    sum_val = warpReduceSum(sum_val);
  }
  if (threadIdx.x == 0) {
    if (sum_val == 0.0f) sum_val = 1e-12f;  // Avoid division by zero
    sdata[0] = sum_val;
  }
  __syncthreads();
  sum_val = sdata[0];

  // Normalize the row elements
  if (vec_possible) {
    int vec_count = D / 4;
    float4* out_vec = reinterpret_cast<float4*>(out);
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      data.x /= sum_val;
      data.y /= sum_val;
      data.z /= sum_val;
      data.w /= sum_val;
      out_vec[row * vec_count + i] = data;
    }
  } else {
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      float data = __ldg(&x[row * D + i]);
      out[row * D + i] = data / sum_val;
    }
  }
}

// Host function that prepares the kernel launch
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected a 2D tensor for normalization.");
  x = x.contiguous();
  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Use vectorized loads if D is a multiple of 4
  bool vec_possible = (D % 4 == 0);
  int threads;
  if (vec_possible) {
    int vec_count = D / 4;
    threads = (vec_count < 1024) ? vec_count : 1024;
  } else {
    threads = (D < 1024) ? D : 1024;
  }
  int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
  int shared_mem_size = num_warps * sizeof(float);

  l1_norm_aligned_kernel<<<N, threads, shared_mem_size>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Aligned __ldg L1 Normalization forward pass (CUDA)");
}
