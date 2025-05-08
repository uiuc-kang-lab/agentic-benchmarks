#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// CUDA kernel implementing L1 normalization with warp shuffle reduction
__global__ void l1_norm_forward_kernel_warp_optimized(const float* __restrict__ x,
                                                       float* __restrict__ out,
                                                       int N,
                                                       int D) {
  int row = blockIdx.x;
  float thread_sum = 0.0f;

  // Check if vectorized (128-bit) loads can be used
  bool vec4_possible = (D % 4 == 0);

  if (vec4_possible) {
    int vec_count = D / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      thread_sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
    }
  } else {
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      float data = __ldg(&x[row * D + col]);
      thread_sum += fabsf(data);
    }
  }

  // Warp-level reduction using shuffle intrinsics
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
  }

  // Write warp reduction result to global memory, and use first thread to sum the results
  if (threadIdx.x % WARP_SIZE == 0) {
    atomicAdd(&out[row], thread_sum);
  }
  __syncthreads();

  float total_sum = out[row];

  // Normalize the row
  if (total_sum > 0.0f) {
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
        float data = __ldg(&x[row * D + col]);
        out[row * D + col] = data / total_sum;
      }
    }
  }
}

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
  x = x.contiguous();

  auto out = torch::zeros({x.size(0)}, x.options());
  int N = x.size(0);
  int D = x.size(1);

  int threads = std::min<int>(1024, D);

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
