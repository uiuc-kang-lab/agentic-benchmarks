#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// This kernel uses one warp (32 threads) per row. It relies solely on warp-level primitives
// (i.e., __shfl_down_sync) for reduction, eliminating the need for shared memory.
// Vectorized loads (using float4) are used when the number of columns D is a multiple of 4.

#define WARP_SIZE 32

__global__ void warp_level_l1_norm_kernel(const float* __restrict__ x,
                                            float* __restrict__ out,
                                            int N,
                                            int D) {
  // Each block processes one row
  int row = blockIdx.x;
  
  // Determine if vectorized loads are possible
  bool use_vec = ((D % 4) == 0) && (D >= 4);
  float sum = 0.0f;

  if (use_vec) {
    // Process data in groups of 4 floats
    int vec_count = D / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += WARP_SIZE) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
    }
  } else {
    for (int i = threadIdx.x; i < D; i += WARP_SIZE) {
      float data = __ldg(&x[row * D + i]);
      sum += fabsf(data);
    }
  }

  // Perform warp-level reduction using __shfl_down_sync
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  // Broadcast the reduced sum (norm) from lane 0 to all lanes
  float norm = __shfl_sync(0xffffffff, sum, 0);
  norm = (norm == 0.0f) ? 1e-12f : norm;

  // Normalize the row elements
  if (use_vec) {
    int vec_count = D / 4;
    float4* out_vec = reinterpret_cast<float4*>(out);
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += WARP_SIZE) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      data.x /= norm;
      data.y /= norm;
      data.z /= norm;
      data.w /= norm;
      out_vec[row * vec_count + i] = data;
    }
  } else {
    for (int i = threadIdx.x; i < D; i += WARP_SIZE) {
      float data = __ldg(&x[row * D + i]);
      out[row * D + i] = data / norm;
    }
  }
}

// Host function to launch the warp-level optimized kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Launch one block per row using exactly one warp (32 threads per block)
  int threads = WARP_SIZE;
  warp_level_l1_norm_kernel<<<N, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), N, D);

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Warp-level optimized L1 Normalization forward pass (CUDA)");
}
