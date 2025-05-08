#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Warp-level reduction using shuffle intrinsics
__device__ inline float warpReduceSum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Block-level reduction using shared memory and warp-level primitives
__device__ float block_reduce_sum(float val, volatile float* sdata) {
  int lane = threadIdx.x % WARP_SIZE;
  int warpid = threadIdx.x / WARP_SIZE;

  // Each warp performs a reduction
  val = warpReduceSum(val);
  if (lane == 0) {
    sdata[warpid] = val;
  }
  __syncthreads();

  // Let the first warp reduce the partial sums
  int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  if (threadIdx.x < numWarps) {
    val = sdata[lane];
    val = warpReduceSum(val);
    if (threadIdx.x == 0) {
      sdata[0] = (val == 0.0f) ? 1e-12f : val;
    }
  }
  __syncthreads();
  return sdata[0];
}

// Device function: Compute partial L1 sum using vectorized float4 loads
__device__ float compute_partial_sum_vectorized(const float* __restrict__ x, int row, int D) {
  float sum = 0.0f;
  int vec_count = D / 4;
  const float4* x_vec = reinterpret_cast<const float4*>(x);
  for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
    float4 data = __ldg(&x_vec[row * vec_count + i]);
    sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
  }
  return sum;
}

// Device function: Compute partial L1 sum using scalar loads
__device__ float compute_partial_sum_scalar(const float* __restrict__ x, int row, int D) {
  float sum = 0.0f;
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    sum += fabsf(__ldg(&x[row * D + col]));
  }
  return sum;
}

// Device function: Normalize a row using vectorized float4 stores
__device__ void normalize_row_vectorized(const float* __restrict__ x, float* __restrict__ out, int row, int D, float norm) {
  int vec_count = D / 4;
  const float4* x_vec = reinterpret_cast<const float4*>(x);
  float4* out_vec = reinterpret_cast<float4*>(out);
  for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
    float4 data = __ldg(&x_vec[row * vec_count + i]);
    data.x /= norm;
    data.y /= norm;
    data.z /= norm;
    data.w /= norm;
    out_vec[row * vec_count + i] = data;
  }
}

// Device function: Normalize a row using scalar stores
__device__ void normalize_row_scalar(const float* __restrict__ x, float* __restrict__ out, int row, int D, float norm) {
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    float val = __ldg(&x[row * D + col]);
    out[row * D + col] = val / norm;
  }
}

// Main kernel: Each block processes one row of the input tensor
__global__ void modular_l1_norm_kernel_refactored(const float* __restrict__ x,
                                                    float* __restrict__ out,
                                                    int N,
                                                    int D) {
  int row = blockIdx.x;
  if (row >= N) return;
  
  extern __shared__ float sdata[];
  
  // Decide on vectorized vs scalar load/store
  bool use_vector = (D % 4 == 0) && (D >= 4);
  float thread_sum = use_vector ? compute_partial_sum_vectorized(x, row, D)
                                  : compute_partial_sum_scalar(x, row, D);
  
  // Block-level reduction to compute the total L1 norm for the row
  float total = block_reduce_sum(thread_sum, sdata);
  
  // Normalize the row elements using the computed norm
  if (use_vector) {
    normalize_row_vectorized(x, out, row, D, total);
  } else {
    normalize_row_scalar(x, out, row, D, total);
  }
}

// Host function that launches the kernel
torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);
  
  // Use vectorized loads/stores if possible
  bool use_vector = (D % 4 == 0) && (D >= 4);
  int threads = use_vector ? ((D / 4) < 1024 ? (D / 4) : 1024) : (D < 1024 ? D : 1024);
  int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
  int shared_mem_size = num_warps * sizeof(float);
  
  // Launch one block per row
  modular_l1_norm_kernel_refactored<<<N, threads, shared_mem_size>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    N,
    D
  );
  
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Modular L1 Normalization forward pass (CUDA) refactored into modular device functions");
}
