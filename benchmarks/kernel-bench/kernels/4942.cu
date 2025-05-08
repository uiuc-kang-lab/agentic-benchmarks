#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel using a two-phase reduction: intra-warp reduction with warp-level primitives
// and inter-warp reduction via shared memory
__global__ void l1_norm_forward_kernel_sharedwarp_opt(const float* __restrict__ x,
                                                       float* __restrict__ out,
                                                       int N,
                                                       int D) {
  __shared__ float sdata[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warpSize = 32;
  const int lane = tid & (warpSize - 1);
  const int warpId = tid / warpSize;
  
  int rowStart = row * D;
  float localSum = 0.0f;
  int nVec = D / 4;
  int rem = D - nVec * 4;
  
  // Process the vectorized portion using float4 loads
  const float4* x_vec = reinterpret_cast<const float4*>(x + rowStart);
  for (int i = tid; i < nVec; i += blockDim.x) {
    float4 v = __ldg(x_vec + i);
    localSum += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
  }
  
  // Process the remaining elements
  int base = nVec * 4;
  for (int i = tid; i < rem; i += blockDim.x) {
    localSum += fabsf(__ldg(x + rowStart + base + i));
  }
  
  // Intra-warp reduction using shfl_down_sync
  unsigned int mask = 0xFFFFFFFF;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    localSum += __shfl_down_sync(mask, localSum, offset);
  }

  // Each warp's first thread stores its reduced sum into shared memory
  if (lane == 0) {
    sdata[warpId] = localSum;
  }
  __syncthreads();
  
  // Final reduction across warp sums
  int numWarps = (blockDim.x + warpSize - 1) / warpSize;
  float finalSum = (tid < numWarps) ? sdata[tid] : 0.0f;
  if (tid < warpSize) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      finalSum += __shfl_down_sync(mask, finalSum, offset);
    }
  }
  if (tid == 0) {
    sdata[0] = fmaxf(finalSum, 1e-12f);  // Avoid division by zero
  }
  __syncthreads();
  float norm = sdata[0];
  
  // Normalize the vectorized portion
  float4* out_vec = reinterpret_cast<float4*>(out + rowStart);
  for (int i = tid; i < nVec; i += blockDim.x) {
    float4 v = __ldg(x_vec + i);
    out_vec[i] = make_float4(v.x / norm, v.y / norm, v.z / norm, v.w / norm);
  }
  
  // Normalize the remainder elements
  for (int i = tid; i < rem; i += blockDim.x) {
    int idx = rowStart + base + i;
    out[idx] = __ldg(x + idx) / norm;
  }
}

// Forward function wrapping the kernel launch
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected a 2D tensor.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Choose number of threads as a multiple of 32 (warp size), up to 1024
  int threads = std::min(1024, D);
  int numWarps = (threads + 31) / 32;
  int sharedMem = numWarps * sizeof(float);

  // Launch one block per row
  l1_norm_forward_kernel_sharedwarp_opt<<<N, threads, sharedMem>>>(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      N,
      D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "L1 Normalization forward pass with internal shared memory and warp-level reduction");
}
