#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Warp-optimized kernel: uses warp-level primitives for reduction, minimizing shared memory usage.
__global__ void l1_norm_forward_kernel_warp_optimized(const float* __restrict__ x,
                                                       float* __restrict__ out,
                                                       int N,
                                                       int D) {
  int row = blockIdx.x;
  float thread_sum = 0.0f;

  // Determine if vectorized loads (float4) can be applied
  bool vec_possible = (D % 4 == 0);

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

  // Final reduction and normalization by the first thread in each warp
  if (threadIdx.x % WARP_SIZE == 0) {
    atomicAdd(&out[row], thread_sum);
  }
  __syncthreads();
  float total_sum = out[row];

  // Normalize the row elements
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

// Host function to launch the warp-optimized CUDA kernel
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

  l1_norm_forward_kernel_warp_optimized<<<N, threads>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Warp-optimized L1 Normalization forward pass (CUDA)");
}