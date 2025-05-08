#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Device function: warp-level reduction using shuffle instructions
__device__ inline float warpReduceSum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Device function: block-level reduction using shared memory
__device__ float blockReduceSum(float val) {
  __shared__ float shared[32]; // supports up to 1024 threads (32 warps)
  int lane = threadIdx.x % WARP_SIZE;
  int wid  = threadIdx.x / WARP_SIZE;
  
  // Each warp fully reduces its own values
  val = warpReduceSum(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  // Let the first warp load all partial sums and reduce them
  if (threadIdx.x < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
    val = shared[lane];
  } else {
    val = 0.0f;
  }
  if (threadIdx.x < WARP_SIZE) {
    val = warpReduceSum(val);
  }
  __syncthreads();
  
  // Thread 0 writes the final sum back to shared memory for broadcasting
  if (threadIdx.x == 0) {
    float total = val;
    total = (total == 0.0f) ? 1e-12f : total;  // avoid division by zero
    shared[0] = total;
  }
  __syncthreads();
  return shared[0];
}

// Device function: Calculate partial L1 norm sum for a row
__device__ float computePartialSum(const float* __restrict__ x, int row, int D, bool use_vec) {
  float sum = 0.0f;
  if (use_vec) {
    // Use vectorized loads when possible (assumes D is a multiple of 4)
    int vec_count = D / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
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

// Global CUDA kernel: Processes one row per block using modular device functions
__global__ void l1_norm_forward_kernel_modular(const float* __restrict__ x,
                                                 float* __restrict__ out,
                                                 int N,
                                                 int D,
                                                 bool use_vec) {
  int row = blockIdx.x;
  if (row < N) {
    // Each thread computes a partial sum over the row
    float partial = computePartialSum(x, row, D, use_vec);
    // Perform block-level reduction to obtain the total L1 norm
    float total = blockReduceSum(partial);
    // Normalize the row elements using the computed L1 norm
    normalizeRow(x, out, row, D, total, use_vec);
  }
}

// Host function to launch the modular CUDA kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Decide whether to use vectorized loads (requires D to be a multiple of 4)
  bool use_vec = (D % 4 == 0);

  // Use min(1024, D) threads per block
  int threads = (D < 1024) ? D : 1024;

  // Launch one block per row
  l1_norm_forward_kernel_modular<<<N, threads>>>(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      N,
      D,
      use_vec
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "L1 Normalization forward pass with modular device functions (CUDA)");
}
