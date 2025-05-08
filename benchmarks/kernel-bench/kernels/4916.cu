#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Optimized kernel using warp-level primitives (__shfl_down_sync) for reduction
// Each block processes one row. Partial sums are computed using vectorized loads and
// then reduced within each warp. Warp leaders write their sums to a small shared memory
// buffer, which is then reduced by thread 0 and broadcast to all threads for normalization.

__global__ void l1_norm_forward_kernel_warp(const float* __restrict__ x,
                                              float* __restrict__ out,
                                              int N,
                                              int D) {
  // Each block processes multiple rows to increase occupancy
  const int ROWS_PER_BLOCK = 4;  // Process 4 rows per block for better occupancy
  int row_base = blockIdx.x * ROWS_PER_BLOCK;
  int tid = threadIdx.x;
  
  // Initialize sums for all rows this block will process
  float sums[ROWS_PER_BLOCK] = {0.0f};
  
  // Use vectorized loads if possible
  int nVec = D / 4; // number of float4 elements
  int rem = D % 4;  // remaining elements
  const float4* x_vec = reinterpret_cast<const float4*>(x + row_start);
  
  // Each thread processes multiple vectorized chunks
  for (int i = threadIdx.x; i < nVec; i += blockDim.x) {
      float4 val = __ldg(x_vec + i);
      sum += fabsf(val.x) + fabsf(val.y) + fabsf(val.z) + fabsf(val.w);
  }
  
  // Process remaining elements
  int offset = nVec * 4;
  for (int j = threadIdx.x; j < rem; j += blockDim.x) {
      float val = __ldg(x + row_start + offset + j);
      sum += fabsf(val);
  }

  // Warp-level reduction using shuffle primitives
  unsigned int lane = threadIdx.x & 31;  // lane index in the warp
  // Reduce within each warp
  for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  // Allocate shared memory for storing warp-level sums
  // The shared memory layout: first numWarps elements for warp sums, then one extra element for
  // the final block sum
  extern __shared__ float shared[]; 
  int numWarps = (blockDim.x + 31) / 32;
  if (lane == 0) {
      shared[threadIdx.x >> 5] = sum;  // store the warp's reduced sum
  }
  __syncthreads();

  // Let thread 0 reduce the warp sums
  float blockSum = 0.0f;
  if (threadIdx.x == 0) {
      for (int i = 0; i < numWarps; i++) {
          blockSum += shared[i];
      }
      // Prevent division by zero
      if (blockSum == 0.0f) {
          blockSum = 1e-12f;
      }
      shared[numWarps] = blockSum;  // store final sum for broadcasting
  }
  __syncthreads();

  // Retrieve the block-level (row) sum
  float totalSum = shared[numWarps];

  // Normalization: each thread normalizes a portion of the row
  // Process vectorized portion
  float4* out_vec = reinterpret_cast<float4*>(out + row_start);
  for (int i = threadIdx.x; i < nVec; i += blockDim.x) {
      float4 v = __ldg(x_vec + i);
      float4 res;
      res.x = v.x / totalSum;
      res.y = v.y / totalSum;
      res.z = v.z / totalSum;
      res.w = v.w / totalSum;
      out_vec[i] = res;
  }
  
  // Process remaining elements scalarly
  for (int j = threadIdx.x; j < rem; j += blockDim.x) {
      int idx = row_start + nVec * 4 + j;
      out[idx] = x[idx] / totalSum;
  }
}


// The forward function
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Choose number of threads per block (max 1024 or D if D is smaller)
  int threads = std::min<int>(1024, D);
  int numWarps = (threads + 31) / 32;
  // Shared memory size: numWarps warp-sum slots + 1 extra for block sum
  int shared_mem_size = (numWarps + 1) * sizeof(float);

  // Launch one block per row
  l1_norm_forward_kernel_warp<<<N, threads, shared_mem_size>>>(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      N,
      D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "L1 Normalization forward pass (Warp-Level Reduction CUDA)");
}
