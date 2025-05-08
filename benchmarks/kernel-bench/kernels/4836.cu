#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Device function for warp-level reduction using shuffle instructions
__device__ inline float warpReduceSum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Device function for block-level reduction using shared memory
__device__ float blockReduceSum(float val) {
  __shared__ float shared[32]; // supports up to 1024 threads (32 warps)
  int lane = threadIdx.x % WARP_SIZE;
  int wid  = threadIdx.x / WARP_SIZE;
  
  // Each warp reduces its values
  val = warpReduceSum(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  
  // Let the first warp reduce the partial sums
  int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  val = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
  if (threadIdx.x < WARP_SIZE) {
    val = warpReduceSum(val);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    // Avoid division by zero
    shared[0] = (val == 0.0f ? 1e-12f : val);
  }
  __syncthreads();
  return shared[0];
}

// Device function: Compute partial sum of absolute values for a row
__device__ float computePartialSum(const float* __restrict__ x, int row, int D, bool use_vec) {
  float sum = 0.0f;
  if (use_vec) {
    int vec_count = D / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      // Use __ldg for cache-friendly loads
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
    }
  } else {
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      sum += fabsf(__ldg(&x[row * D + col]));
    }
  }
  return sum;
}

// Device function: Normalize a row using the computed L1 norm
__device__ void normalizeRow(const float* __restrict__ x, float* __restrict__ out, int row, int D, float norm, bool use_vec) {
  if (use_vec) {
    int vec_count = D / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* out_vec = reinterpret_cast<float4*>(out);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 data = __ldg(&x_vec[row * vec_count + i]);
      data.x = data.x / norm;
      data.y = data.y / norm;
      data.z = data.z / norm;
      data.w = data.w / norm;
      out_vec[row * vec_count + i] = data;
    }
  } else {
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      float val = __ldg(&x[row * D + col]);
      out[row * D + col] = val / norm;
    }
  }
}

// Combined CUDA kernel: Each block processes one row of the tensor
__global__ void l1_norm_forward_kernel_combined(const float* __restrict__ x,
                                                  float* __restrict__ out,
                                                  int N,
                                                  int D,
                                                  bool use_vec) {
  int row = blockIdx.x;
  if (row < N) {
    // Compute the partial L1 norm sum using vectorized loads if enabled
    float partial = computePartialSum(x, row, D, use_vec);
    // Perform block-level reduction to get total L1 norm.
    float total = blockReduceSum(partial);
    // Normalize the row elements by the L1 norm
    normalizeRow(x, out, row, D, total, use_vec);
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
  
  // Enable vectorized loads/stores if D is a multiple of 4 and large enough
  bool use_vec = (D % 4 == 0) && (D >= 4);
  
  // Choose threads per block: if vectorized, work on D/4 elements per row; otherwise D elements
  int threads = use_vec ? ((D / 4) < 1024 ? (D / 4) : 1024) : (D < 1024 ? D : 1024);
  
  // Launch one block per row
  l1_norm_forward_kernel_combined<<<N, threads>>>(x.data_ptr<float>(),
                                                    out.data_ptr<float>(),
                                                    N,
                                                    D,
                                                    use_vec);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Combined L1 Normalization forward pass with vectorized loads (CUDA)");
}
