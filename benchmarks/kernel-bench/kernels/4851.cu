#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

__device__ inline float warpReduceSum(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void l1_norm_kernel(const float* __restrict__ x,
                              float* __restrict__ out,
                              int N,
                              int D) {
  extern __shared__ float s_norm[];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  float sum = 0.0f;

  const bool use_vec = (D % 4 == 0);

  if (use_vec) {
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    const int vec_size = D / 4;
    for (int i = tid; i < vec_size; i += blockDim.x) {
      float4 chunk = __ldg(&x_vec[row * vec_size + i]);
      sum += fabsf(chunk.x) + fabsf(chunk.y) + fabsf(chunk.z) + fabsf(chunk.w);
    }
    // Handle remainder elements
    for (int i = D - (D % 4) + tid; i < D; i += blockDim.x)
      sum += fabsf(__ldg(&x[row * D + i]));
  } else {
    for (int i = tid; i < D; i += blockDim.x)
      sum += fabsf(__ldg(&x[row * D + i]));
  }

  sum = warpReduceSum(sum);

  if (tid % WARP_SIZE == 0)
    s_norm[tid/WARP_SIZE] = sum;
  __syncthreads();

  if (tid < WARP_SIZE) {
    sum = (tid < (blockDim.x + WARP_SIZE-1)/WARP_SIZE) ? s_norm[tid] : 0.0f;
    sum = warpReduceSum(sum);
  }

  __shared__ float total_sum;
  if (tid == 0) {
    total_sum = (sum == 0.0f) ? 1e-12f : sum;
  }
  __syncthreads();

  if (use_vec) {
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* out_vec = reinterpret_cast<float4*>(out);
    const int vec_size = D / 4;
    for (int i = tid; i < vec_size; i += blockDim.x) {
      float4 chunk = __ldg(&x_vec[row * vec_size + i]);
      chunk.x /= total_sum;
      chunk.y /= total_sum;
      chunk.z /= total_sum;
      chunk.w /= total_sum;
      out_vec[row * vec_size + i] = chunk;
    }
    // Handle remainder elements
    for (int i = D - (D % 4) + tid; i < D; i += blockDim.x)
      out[row * D + i] = __ldg(&x[row * D + i]) / total_sum;
  } else {
    for (int i = tid; i < D; i += blockDim.x)
      out[row * D + i] = __ldg(&x[row * D + i]) / total_sum;
  }
}

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  const int N = x.size(0);
  const int D = x.size(1);

  const int max_threads = 1024;
  const int threads = (D >= 4 && (D % 4 == 0)) ? 
                     std::min(max_threads, (D + 3) / 4) : 
                     std::min(max_threads, D);
  const int smem = ((threads + WARP_SIZE-1)/WARP_SIZE) * sizeof(float);

  l1_norm_kernel<<<N, threads, smem>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Atomic-free L1 Normalization (CUDA)");
}