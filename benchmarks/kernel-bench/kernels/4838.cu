#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Optimized warp reduction using shuffle
__device__ inline float warpReduceSum(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Vectorized data loading with L1 norm computation
__device__ float computeVectorizedSum(const float4* x_vec, int row, int vec_count) {
  float sum = 0.0f;
  #pragma unroll 4
  for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
    float4 data = __ldg(&x_vec[row * vec_count + i]);
    sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
  }
  return sum;
}

// Optimized block reduction with minimal syncs
__device__ float blockReduceSum(float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val);
  
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared[lane] : 0.0f;
  
  if (threadIdx.x < WARP_SIZE) {
    val = warpReduceSum(val);
    if (threadIdx.x == 0) {
      shared[0] = (val == 0.0f) ? 1e-12f : val;
    }
  }
  __syncthreads();
  return shared[0];
}

__global__ void l1_norm_kernel_optimized(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int N,
                                       int D) {
  const int row = blockIdx.x;
  float partial_sum = 0.0f;

  if (D >= 4) {
    // Use vectorized loads for aligned data
    const int vec_count = D / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    partial_sum = computeVectorizedSum(x_vec, row, vec_count);

    // Handle remaining elements
    #pragma unroll
    for (int col = D - (D % 4) + threadIdx.x; col < D; col += blockDim.x) {
      partial_sum += fabsf(__ldg(&x[row * D + col]));
    }
  } else {
    // Standard processing for small D
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      partial_sum += fabsf(__ldg(&x[row * D + col]));
    }
  }

  const float norm = blockReduceSum(partial_sum);

  // Vectorized output writing
  if (D >= 4) {
    const int vec_count = D / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* out_vec = reinterpret_cast<float4*>(out);
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      data.x /= norm;
      data.y /= norm;
      data.z /= norm;
      data.w /= norm;
      out_vec[row * vec_count + i] = data;
    }

    // Handle remaining elements
    for (int col = D - (D % 4) + threadIdx.x; col < D; col += blockDim.x) {
      out[row * D + col] = x[row * D + col] / norm;
    }
  } else {
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      out[row * D + col] = x[row * D + col] / norm;
    }
  }
}

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  const int N = x.size(0);
  const int D = x.size(1);
  const int threads = std::min(1024, ((D + 3) / 4) * 4); // Align thread count to vector size

  l1_norm_kernel_optimized<<<N, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), N, D);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized L1 Normalization (CUDA)");
}