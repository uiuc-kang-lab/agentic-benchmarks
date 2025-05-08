#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Optimized kernel with improved thread and block indexing
__global__ void l1_norm_forward_kernel_optimized(const float* __restrict__ x,
                                                  float* __restrict__ out,
                                                  int N,
                                                  int D) {
  int row = blockIdx.x * blockDim.y + threadIdx.y; // Map each block to a row
  if (row >= N) return;

  float thread_sum = 0.0f;
  bool vec_possible = (D % 4 == 0);

  if (vec_possible) {
    int vec_count = D / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      thread_sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
    }
  } else {
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      float data = __ldg(&x[row * D + i]);
      thread_sum += fabsf(data);
    }
  }

  // Warp-level reduction using shuffle intrinsics
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
  }

  // Use atomic operations to accumulate results from each warp
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

// Host function to launch the optimized CUDA kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this normalization.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  bool vec_possible = (D % 4 == 0);
  int threads = vec_possible ? 256 : 256; // Use 256 threads per block
  int blocks = (N + threads - 1) / threads; // Calculate number of blocks needed

  dim3 blockDim(threads, 1);
  dim3 gridDim(blocks);

  l1_norm_forward_kernel_optimized<<<gridDim, blockDim>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized L1 Normalization forward pass (CUDA)");
}