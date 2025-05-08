#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Combined CUDA kernel with vectorized loads/stores and optimized warp reduction
// Performs L1 normalization (row-wise) using __ldg(), float4 vectorization when possible, and minimal synchronizations.
__global__ void l1_norm_forward_kernel_combined(const float* __restrict__ x,
                                                  float* __restrict__ out,
                                                  int N,
                                                  int D) {
  int row = blockIdx.x;
  float thread_sum = 0.0f;
  
  // Check if we can use vectorized (float4) loads/stores
  bool vec4_possible = (D % 4 == 0);
  int vec_count = vec4_possible ? (D / 4) : 0;

  // Accumulate partial sum using either vectorized or scalar loads
  if (vec4_possible) {
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      thread_sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
    }
  } else {
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      float val = __ldg(&x[row * D + col]);
      thread_sum += fabsf(val);
    }
  }

  // Perform warp-level reduction using shuffle intrinsics
  unsigned int lane = threadIdx.x % WARP_SIZE;
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
  }

  // Each warp writes its result to shared memory
  extern __shared__ float sdata[];
  int warp_id = threadIdx.x / WARP_SIZE;
  if (lane == 0) {
    sdata[warp_id] = thread_sum;
  }
  __syncthreads();

  // Final reduction across warps performed by thread 0
  if (threadIdx.x == 0) {
    float total_sum = 0.0f;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    for (int i = 0; i < num_warps; i++) {
      total_sum += sdata[i];
    }
    // Avoid division by zero
    sdata[0] = (total_sum == 0.0f ? 1e-12f : total_sum);
  }
  __syncthreads();

  float total_sum = sdata[0];

  // Normalize the row elements using the computed total sum
  if (vec4_possible) {
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
  } else {
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      float val = __ldg(&x[row * D + col]);
      out[row * D + col] = val / total_sum;
    }
  }
}

// Host function to launch the combined CUDA kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for L1 normalization.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Pick number of threads per block up to 1024, but not exceeding D
  int threads = (D < 1024) ? D : 1024;
  int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
  int shared_mem_size = num_warps * sizeof(float);

  l1_norm_forward_kernel_combined<<<N, threads, shared_mem_size>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Combined L1 Normalization forward pass with vectorized loads/stores (CUDA)");
}
