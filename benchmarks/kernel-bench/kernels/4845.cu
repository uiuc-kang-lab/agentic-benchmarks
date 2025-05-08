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

// Modular block reduction: reduce values across the block using shared memory
__device__ float get_block_sum(float thread_total, volatile float* sdata) {
  int lane = threadIdx.x % WARP_SIZE;
  int warpId = threadIdx.x / WARP_SIZE;
  
  // Each warp performs a warp-level reduction
  thread_total = warpReduceSum(thread_total);

  // Write reduced value of each warp to shared memory
  if (lane == 0) {
    sdata[warpId] = thread_total;
  }
  __syncthreads();

  // First warp loads all warp sums and reduces them
  float block_total = 0.0f;
  if (threadIdx.x < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
    block_total = sdata[lane];
    block_total = warpReduceSum(block_total);
  }
  __syncthreads();

  // Thread 0 writes the final sum back to shared memory and ensures non-zero norm
  if (threadIdx.x == 0) {
    sdata[0] = (block_total == 0.0f) ? 1e-12f : block_total;
  }
  __syncthreads();
  
  return sdata[0];
}

// Device function for vectorized partial sum computation using float4
__device__ float compute_partial_sum_vec(const float* __restrict__ x, int row, int vec_count) {
  float sum = 0.0f;
  const float4* x_vec = reinterpret_cast<const float4*>(x);
  for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
    float4 data = __ldg(&x_vec[row * vec_count + i]);
    sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
  }
  return sum;
}

// Device function for scalar partial sum computation
__device__ float compute_partial_sum_scalar(const float* __restrict__ x, int row, int D) {
  float sum = 0.0f;
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    sum += fabsf(__ldg(&x[row * D + col]));
  }
  return sum;
}

// Device function for vectorized normalization using float4
__device__ void normalize_row_vec(const float* __restrict__ x, float* __restrict__ out, int row, int vec_count, float norm) {
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
}

// Device function for scalar normalization
__device__ void normalize_row_scalar(const float* __restrict__ x, float* __restrict__ out, int row, int D, float norm) {
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    float value = __ldg(&x[row * D + col]);
    out[row * D + col] = value / norm;
  }
}

// Modular CUDA kernel for L1 normalization
// Each block processes one row of the input tensor
__global__ void modular_l1_norm_kernel(const float* __restrict__ x,
                                         float* __restrict__ out,
                                         int N,
                                         int D) {
  int row = blockIdx.x;
  if (row >= N) return;
  
  // Shared memory for block-level reduction
  extern __shared__ float sdata[];
  
  // Decide if we can use vectorized loads/stores (float4) when D is a multiple of 4
  bool vec_possible = (D % 4 == 0) && (D >= 4);
  float thread_sum = 0.0f;
  
  if (vec_possible) {
    int vec_count = D / 4;
    thread_sum = compute_partial_sum_vec(x, row, vec_count);
  } else {
    thread_sum = compute_partial_sum_scalar(x, row, D);
  }
  
  // Perform block-level reduction to get the L1 norm for the row
  float total = get_block_sum(thread_sum, sdata);
  
  // Normalize the row elements using the computed norm
  if (vec_possible) {
    int vec_count = D / 4;
    normalize_row_vec(x, out, row, vec_count, total);
  } else {
    normalize_row_scalar(x, out, row, D, total);
  }
}

// Host function to launch the modular L1 normalization kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
  x = x.contiguous();
  
  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);
  
  bool vec_possible = (D % 4 == 0) && (D >= 4);
  int threads = 0;
  if (vec_possible) {
    int vec_count = D / 4;
    threads = (vec_count < 1024) ? vec_count : 1024;
  } else {
    threads = (D < 1024) ? D : 1024;
  }
  
  // Allocate shared memory: one float per warp
  int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
  int shared_mem_size = num_warps * sizeof(float);
  
  // Launch one block per row
  modular_l1_norm_kernel<<<N, threads, shared_mem_size>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );
  
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Modular L1 Normalization forward pass (CUDA)");
}
