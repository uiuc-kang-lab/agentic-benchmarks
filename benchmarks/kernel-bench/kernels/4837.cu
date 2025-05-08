/*
Combined CUDA Kernel for L1 Normalization
Combines vectorized __ldg accesses and float4 loads/stores (when possible) with efficient warp-level and shared-memory reduction. 
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Combined kernel: uses vectorized loads/stores if D is a multiple of 4, otherwise falls back to scalar loads.
__global__ void l1_norm_forward_kernel_combined(const float* __restrict__ x,
                                                 float* __restrict__ out,
                                                 int N,
                                                 int D) {
  int row = blockIdx.x;

  // Determine if vectorized loads (float4) can be applied
  bool vec_possible = (D % 4 == 0);
  float thread_sum = 0.0f;

  if (vec_possible) {
    int vec_count = D / 4;  // number of float4 elements
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      thread_sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
    }
  } else {
    // Fallback to scalar loads if not vectorizable
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      float data = __ldg(&x[row * D + i]);
      thread_sum += fabsf(data);
    }
  }

  // Warp-level reduction using shuffle intrinsics
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
  }

  // Shared memory reduction across warps
  extern __shared__ float sdata[];
  int warp_id = threadIdx.x / WARP_SIZE;
  if (threadIdx.x % WARP_SIZE == 0) {
    sdata[warp_id] = thread_sum;
  }
  __syncthreads();

  // Final reduction by thread 0 in the block
  if (threadIdx.x == 0) {
    float total_sum = 0.0f;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    for (int i = 0; i < num_warps; i++) {
      total_sum += sdata[i];
    }
    // Avoid division by zero
    if (total_sum == 0.0f) {
      total_sum = 1e-12f;
    }
    sdata[0] = total_sum;
  }
  __syncthreads();
  float total_sum = sdata[0];

  // Normalize the row elements by dividing with the computed L1 norm
  if (vec_possible) {
    int vec_count = D / 4;
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
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      float data = __ldg(&x[row * D + i]);
      out[row * D + i] = data / total_sum;
    }
  }
}

// Host function to launch the combined CUDA kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this normalization.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Determine thread block size based on whether vectorized loads can be used
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

  l1_norm_forward_kernel_combined<<<N, threads, shared_mem_size>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Combined vectorized L1 Normalization forward pass (CUDA)");
}
