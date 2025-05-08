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

// CUDA kernel that performs L1 normalization on a 2D tensor row by row
// Ensures memory coalescing by having each thread read/write consecutive elements
// If D is a multiple of 4, vectorized loads (float4) are used for aligned accesses
__global__ void coalesced_l1_norm_kernel(const float* __restrict__ x,
                                           float* __restrict__ out,
                                           int N, int D) {
  int row = blockIdx.x;
  if (row >= N) return;

  // Shared memory for block reduction
  extern __shared__ float sdata[];

  // Decide if we can use vectorized (float4) loads/stores
  bool use_vec = ((D % 4) == 0) && (D >= 4);
  float thread_sum = 0.0f;

  if (use_vec) {
    int vec_count = D / 4;  // number of float4 elements
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    // Each thread loads contiguous float4 values for coalesced access
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      thread_sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
    }
  } else {
    // Fallback to scalar loads
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      thread_sum += fabsf(__ldg(&x[row * D + i]));
    }
  }

  // Intra-warp reduction
  thread_sum = warpReduceSum(thread_sum);

  int lane = threadIdx.x & (WARP_SIZE - 1);
  int warpId = threadIdx.x / WARP_SIZE;

  // Write reduced value of each warp to shared memory
  if (lane == 0) {
    sdata[warpId] = thread_sum;
  }
  __syncthreads();

  // Let the first warp perform the final reduction
  float total_sum = 0.0f;
  if (threadIdx.x < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
    total_sum = sdata[lane];
    total_sum = warpReduceSum(total_sum);
  }
  __syncthreads();

  // Ensure non-zero norm
  if (threadIdx.x == 0) {
    if (total_sum == 0.0f) total_sum = 1e-12f;
    sdata[0] = total_sum;
  }
  __syncthreads();
  total_sum = sdata[0];

  // Normalize the row elements with coalesced global memory accesses
  if (use_vec) {
    int vec_count = D / 4;
    float4* out_vec = reinterpret_cast<float4*>(out);
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      data.x = data.x / total_sum;
      data.y = data.y / total_sum;
      data.z = data.z / total_sum;
      data.w = data.w / total_sum;
      out_vec[row * vec_count + i] = data;
    }
  } else {
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      float value = __ldg(&x[row * D + i]);
      out[row * D + i] = value / total_sum;
    }
  }
}

// Host function to launch the kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Decide thread count based on vectorization possibility
  bool use_vec = ((D % 4) == 0) && (D >= 4);
  int element_count = use_vec ? (D / 4) : D;
  int threads = (element_count < 1024) ? element_count : 1024;
  int numWarps = (threads + WARP_SIZE - 1) / WARP_SIZE;
  int shared_mem_size = numWarps * sizeof(float);

  // Launch one block per row
  coalesced_l1_norm_kernel<<<N, threads, shared_mem_size>>>(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      N, D);

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Coalesced L1 Normalization (CUDA)");
}
