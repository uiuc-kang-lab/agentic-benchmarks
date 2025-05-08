#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

__device__ inline float warp_reduce(float val) {
  #pragma unroll
  for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) 
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void l1_norm_optimized(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  int N,
                                  int D) {
  extern __shared__ float smem[];
  
  const int row = blockIdx.x;
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;
  
  float sum = 0.0f;
  
  // Vectorized load when D aligned to 4 elements
  if (D % 4 == 0) {
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    const int vecD = D / 4;
    for (int i = threadIdx.x; i < vecD; i += blockDim.x) {
      float4 v = __ldg(&x_vec[row*vecD + i]);
      sum += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
    }
  } else { // Scalar fallback
    for (int i = threadIdx.x; i < D; i += blockDim.x) 
      sum += fabsf(__ldg(x + row*D + i));
  }

  // Warp-level reduction
  sum = warp_reduce(sum);

  // Store warp sums to shared memory
  if (lane == 0)
    smem[warp_id] = sum;
  __syncthreads();

  // Final reduction by first warp
  if (warp_id == 0) {
    sum = (lane < blockDim.x/WARP_SIZE) ? smem[lane] : 0.0f;
    sum = warp_reduce(sum);
    
    // Handle zero sum and store final norm
    if (lane == 0) {
      sum = fmaxf(sum, 1e-12f);
      smem[0] = sum;
    }
  }
  __syncthreads();
  
  const float norm = smem[0];

  // Normalization with warp-strided writes
  if (D % 4 == 0) {
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* out_vec = reinterpret_cast<float4*>(out);
    const int vecD = D / 4;
    for (int i = threadIdx.x; i < vecD; i += blockDim.x) {
      float4 v = __ldg(&x_vec[row*vecD + i]);
      v.x /= norm; v.y /= norm; v.z /= norm; v.w /= norm;
      out_vec[row*vecD + i] = v;
    }
  } else {
    for (int i = threadIdx.x; i < D; i += blockDim.x) 
      out[row*D + i] = __ldg(x + row*D + i) / norm;
  }
}

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "Input must be 2D");
  x = x.contiguous();
  
  const int N = x.size(0);
  const int D = x.size(1);
  auto out = torch::empty_like(x);

  // Configure threads to match vector size when possible
  int threads = (D % 4 == 0) ? std::min(1024, (D/4 + 31)/32*32) 
                             : std::min(1024, (D + 31)/32*32);
  int warps_per_block = (threads + WARP_SIZE-1)/WARP_SIZE;
  
  l1_norm_optimized<<<N, threads, warps_per_block*sizeof(float)>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized L1 Normalization");
}