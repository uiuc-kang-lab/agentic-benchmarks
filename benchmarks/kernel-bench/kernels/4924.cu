#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void l1_norm_forward_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int N,
                                       int D) {
  extern __shared__ float sdata[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_size = 32;
  const int lane_id = tid % warp_size;
  const int warp_id = tid / warp_size;
  
  // Define tile size for shared memory
  const int TILE_SIZE = 256;  // Adjust based on shared memory size
  __shared__ float4 tile[TILE_SIZE/4];
  
  float sum = 0.0f;
  const float4* x_vec = reinterpret_cast<const float4*>(x + row * D);
  const int nVec = D / 4;
  const int rem = D % 4;
  
  // Process data in tiles
  for (int tile_start = 0; tile_start < nVec; tile_start += TILE_SIZE/4) {
    const int tile_end = min(tile_start + TILE_SIZE/4, nVec);
    
    // Load tile into shared memory
    for (int i = tid; i < tile_end - tile_start; i += blockDim.x) {
      tile[i] = x_vec[tile_start + i];
    }
    __syncthreads();
    
    // Process tile from shared memory
    #pragma unroll
    for (int i = tid; i < tile_end - tile_start; i += blockDim.x) {
      float4 val = tile[i];
      sum += fabsf(val.x) + fabsf(val.y) + fabsf(val.z) + fabsf(val.w);
    }
    __syncthreads();
  }

  // Handle remainder elements
  const int base = nVec * 4;
  #pragma unroll
  for (int i = tid; i < rem; i += blockDim.x) {
    sum += fabsf(__ldg(x + row * D + base + i));
  }

  // Warp-level reduction
  for (int offset = warp_size / 2; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);

  if (lane_id == 0)
    sdata[warp_id] = sum;
  __syncthreads();

  // Final block reduction
  if (warp_id == 0) {
    sum = lane_id < blockDim.x / warp_size ? sdata[lane_id] : 0.0f;
    for (int offset = blockDim.x / warp_size / 2; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (tid == 0)
      sdata[0] = fmaxf(sum, 1e-12f);
  }
  __syncthreads();

  // Coalesced vectorized stores with unrolling
  float4* out_vec = reinterpret_cast<float4*>(out + row * D);
  const float norm = sdata[0];
  
  #pragma unroll
  for (int i = tid; i < nVec; i += blockDim.x) {
    float4 val = __ldg(x_vec + i);
    out_vec[i] = make_float4(val.x / norm, val.y / norm, val.z / norm, val.w / norm);
  }

  // Coalesced remainder stores with unrolling
  #pragma unroll
  for (int i = tid; i < rem; i += blockDim.x) {
    out[row * D + base + i] = __ldg(x + row * D + base + i) / norm;
  }
}

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  const int N = x.size(0);
  const int D = x.size(1);

  const int warp_size = 32;
  const int threads = std::min<int>(1024, (D + 3) / 4 * warp_size);
  const int shared_mem = (threads / warp_size) * sizeof(float);

  l1_norm_forward_kernel<<<N, threads, shared_mem>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "L1 Normalization with coalesced memory access and unrolled loops");
}