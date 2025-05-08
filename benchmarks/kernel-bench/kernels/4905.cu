#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Optimized kernel using __ldg() for read-only accesses and 128-bit vectorized loads/stores.
__global__ void l1_norm_forward_kernel_opt(const float* __restrict__ x,
                                            float* __restrict__ out,
                                            int N,
                                            int D) {
  extern __shared__ float sdata[];
  int row = blockIdx.x;
  int row_start = row * D;
  float sum = 0.0f;

  // Process elements in groups of 4 (128-bit) if possible
  int nVec = D / 4;       // number of vectorized (float4) elements
  int rem = D % 4;        // remaining elements

  // Cast to float4 pointer for aligned accesses
  const float4* x_vec = reinterpret_cast<const float4*>(x + row_start);

  // Each thread processes multiple vectorized chunks
  for (int i = threadIdx.x; i < nVec; i += blockDim.x) {
      // Use __ldg() for read-only global load
      float4 val = __ldg(x_vec + i);
      sum += fabsf(val.x) + fabsf(val.y) + fabsf(val.z) + fabsf(val.w);
  }

  // Process remaining elements that don't make up a full float4
  int vec_elems = nVec * 4;
  for (int j = threadIdx.x; j < rem; j += blockDim.x) {
      float val = __ldg(x + row_start + vec_elems + j);
      sum += fabsf(val);
  }

  // Reduction: store each thread's partial sum in shared memory
  sdata[threadIdx.x] = sum;
  __syncthreads();

  // Reduce the partial sums in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
          sdata[threadIdx.x] += sdata[threadIdx.x + stride];
      }
      __syncthreads();
  }

  // Normalize the row: avoid division by zero
  float total_sum = sdata[0];
  if (threadIdx.x == 0 && total_sum == 0.0f) {
      total_sum = 1e-12f;
      sdata[0] = total_sum;
  }
  __syncthreads();
  total_sum = sdata[0];

  // Write the normalized results back, using vectorized stores when possible
  float4* out_vec = reinterpret_cast<float4*>(out + row_start);
  for (int i = threadIdx.x; i < nVec; i += blockDim.x) {
      float4 val = __ldg(x_vec + i);
      float4 res;
      res.x = val.x / total_sum;
      res.y = val.y / total_sum;
      res.z = val.z / total_sum;
      res.w = val.w / total_sum;
      out_vec[i] = res;
  }

  // Process any remaining elements scalarly
  for (int j = threadIdx.x; j < rem; j += blockDim.x) {
      int idx = row_start + nVec * 4 + j;
      out[idx] = x[idx] / total_sum;
  }
}

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Determine number of threads per block (limit to 1024 and not more than D)
  int threads = std::min<int>(1024, D);
  int shared_mem_size = threads * sizeof(float);

  l1_norm_forward_kernel_opt<<<N, threads, shared_mem_size>>>(
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
