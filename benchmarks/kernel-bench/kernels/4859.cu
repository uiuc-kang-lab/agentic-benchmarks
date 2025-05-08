#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Unrolled L1 Normalization Kernel with manual loop unrolling
__global__ void unroll_l1_norm_kernel(const float* __restrict__ x,
                                        float* __restrict__ out,
                                        int N,
                                        int D) {
  int row = blockIdx.x;
  if (row >= N) return;

  float thread_sum = 0.0f;
  // Use vectorized loads (float4) only if D is a multiple of 4 and at least 4
  bool vec_possible = ((D % 4) == 0) && (D >= 4);

  if (vec_possible) {
    int vec_count = D / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    int unroll_bound = (vec_count / 4) * 4;  // process in chunks of 4
    
    // Manually unroll the loop by a factor of 4
    for (int i = threadIdx.x; i < unroll_bound; i += blockDim.x) {
      #pragma unroll 4
      {
        float4 v0 = __ldg(&x_vec[row * vec_count + i]);
        float4 v1 = __ldg(&x_vec[row * vec_count + i + 1]);
        float4 v2 = __ldg(&x_vec[row * vec_count + i + 2]);
        float4 v3 = __ldg(&x_vec[row * vec_count + i + 3]);
        thread_sum += fabsf(v0.x) + fabsf(v0.y) + fabsf(v0.z) + fabsf(v0.w)
                    + fabsf(v1.x) + fabsf(v1.y) + fabsf(v1.z) + fabsf(v1.w)
                    + fabsf(v2.x) + fabsf(v2.y) + fabsf(v2.z) + fabsf(v2.w)
                    + fabsf(v3.x) + fabsf(v3.y) + fabsf(v3.z) + fabsf(v3.w);
      }
    }
    // Handle any remaining vectorized elements
    for (int i = ((vec_count / 4) * 4) + threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 v = __ldg(&x_vec[row * vec_count + i]);
      thread_sum += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
    }
  } else {
    // Scalar path with manual unrolling when possible
    int r = D / 4;
    int col_bound = r * 4;
    for (int i = threadIdx.x; i < r; i += blockDim.x) {
      int col = i * 4;
      #pragma unroll 4
      {
        float d0 = __ldg(&x[row * D + col]);
        float d1 = __ldg(&x[row * D + col + 1]);
        float d2 = __ldg(&x[row * D + col + 2]);
        float d3 = __ldg(&x[row * D + col + 3]);
        thread_sum += fabsf(d0) + fabsf(d1) + fabsf(d2) + fabsf(d3);
      }
    }
    // Process the remaining elements
    for (int i = col_bound + threadIdx.x; i < D; i += blockDim.x) {
      float d = __ldg(&x[row * D + i]);
      thread_sum += fabsf(d);
    }
  }

  // Warp-level reduction using shuffle with manual unroll
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
  }

  // Shared memory reduction across warps
  extern __shared__ float sdata[];
  int lane = threadIdx.x % WARP_SIZE;
  int warpId = threadIdx.x / WARP_SIZE;
  if (lane == 0) {
    sdata[warpId] = thread_sum;
  }
  __syncthreads();

  // Final reduction among warp sums
  float norm = 0.0f;
  int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  if (threadIdx.x < numWarps) {
    norm = sdata[lane];
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      norm += __shfl_down_sync(0xffffffff, norm, offset);
    }
    if (threadIdx.x == 0) {
      sdata[0] = (norm == 0.0f ? 1e-12f : norm);
    }
  }
  __syncthreads();
  norm = sdata[0];

  // Normalization phase
  if (vec_possible) {
    int vec_count = D / 4;
    float4* out_vec = reinterpret_cast<float4*>(out);
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    int unroll_bound = (vec_count / 4) * 4;
    for (int i = threadIdx.x; i < unroll_bound; i += blockDim.x) {
      #pragma unroll 4
      {
        float4 v0 = __ldg(&x_vec[row * vec_count + i]);
        float4 v1 = __ldg(&x_vec[row * vec_count + i + 1]);
        float4 v2 = __ldg(&x_vec[row * vec_count + i + 2]);
        float4 v3 = __ldg(&x_vec[row * vec_count + i + 3]);
        v0.x /= norm; v0.y /= norm; v0.z /= norm; v0.w /= norm;
        v1.x /= norm; v1.y /= norm; v1.z /= norm; v1.w /= norm;
        v2.x /= norm; v2.y /= norm; v2.z /= norm; v2.w /= norm;
        v3.x /= norm; v3.y /= norm; v3.z /= norm; v3.w /= norm;
        out_vec[row * vec_count + i]     = v0;
        out_vec[row * vec_count + i + 1] = v1;
        out_vec[row * vec_count + i + 2] = v2;
        out_vec[row * vec_count + i + 3] = v3;
      }
    }
    for (int i = ((vec_count / 4) * 4) + threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 v = __ldg(&x_vec[row * vec_count + i]);
      v.x /= norm; v.y /= norm; v.z /= norm; v.w /= norm;
      out_vec[row * vec_count + i] = v;
    }
  } else {
    int r = D / 4;
    int col_bound = r * 4;
    for (int i = threadIdx.x; i < r; i += blockDim.x) {
      int col = i * 4;
      #pragma unroll 4
      {
        float d0 = __ldg(&x[row * D + col]);
        float d1 = __ldg(&x[row * D + col + 1]);
        float d2 = __ldg(&x[row * D + col + 2]);
        float d3 = __ldg(&x[row * D + col + 3]);
        int base = row * D + col;
        out[base]     = d0 / norm;
        out[base + 1] = d1 / norm;
        out[base + 2] = d2 / norm;
        out[base + 3] = d3 / norm;
      }
    }
    for (int i = col_bound + threadIdx.x; i < D; i += blockDim.x) {
      float d = __ldg(&x[row * D + i]);
      out[row * D + i] = d / norm;
    }
  }
}

// Host function to launch the unrolled CUDA kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);
  bool vec_possible = ((D % 4) == 0) && (D >= 4);

  int threads = 0;
  if (vec_possible) {
    int vec_count = D / 4;
    threads = (vec_count < 1024) ? vec_count : 1024;
  } else {
    threads = (D < 1024) ? D : 1024;
  }

  int numWarps = (threads + WARP_SIZE - 1) / WARP_SIZE;
  int shared_mem = numWarps * sizeof(float);

  // Launch one block per row
  unroll_l1_norm_kernel<<<N, threads, shared_mem>>>(
      x.data_ptr<float>(), out.data_ptr<float>(), N, D);

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Unrolled L1 Normalization forward pass (CUDA)");
}
