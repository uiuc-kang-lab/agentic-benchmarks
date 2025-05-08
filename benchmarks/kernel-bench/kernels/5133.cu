#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <cmath>

// Warp-level reduction using shuffle intrinsic
template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
    // Assuming warpSize is 32
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Optimized LayerNorm forward kernel combining chunked loads and warp-level reduction
template <typename scalar_t>
__global__ void layernorm_forward_kernel_optimized(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block processes one instance
  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;

  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  // Local accumulators for sum and sum of squares
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;

  // Use chunked loading for coalesced access and loop unrolling
  constexpr int CHUNK_SIZE = 4;  // Process 4 elements per thread per iteration
  int num_chunks = (normalized_size + blockDim.x * CHUNK_SIZE - 1) / (blockDim.x * CHUNK_SIZE);

  for (int c = 0; c < num_chunks; c++) {
    int base_idx = c * blockDim.x * CHUNK_SIZE + tid;
    #pragma unroll
    for (int j = 0; j < CHUNK_SIZE; j++) {
      int idx = base_idx + j * blockDim.x;
      if (idx < normalized_size) {
         accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
         local_sum += val;
         local_sum_sq += val * val;
      }
    }
  }

  // Perform warp-level reduction on each thread's local sum
  accscalar_t warp_sum = warpReduceSum(local_sum);
  accscalar_t warp_sum_sq = warpReduceSum(local_sum_sq);

  // Allocate shared memory to hold each warp's reduction results
  extern __shared__ accscalar_t shared_mem[];
  // Number of warps per block (assume warp size is 32)
  int numWarps = blockDim.x / 32;
  accscalar_t* s_sum = shared_mem;             // to store warp sums
  accscalar_t* s_sum_sq = shared_mem + numWarps; // to store warp sums of squares

  int lane = tid & 31;
  int warp_id = tid >> 5;
  if (lane == 0) {
      s_sum[warp_id] = warp_sum;
      s_sum_sq[warp_id] = warp_sum_sq;
  }
  __syncthreads();

  // Final reduction among warp results performed by the first warp
  accscalar_t total_sum = 0;
  accscalar_t total_sum_sq = 0;
  if (tid < numWarps) {
      total_sum = s_sum[tid];
      total_sum_sq = s_sum_sq[tid];
  }
  if (tid < 32) { // Only the first warp carries out the final reduction
      total_sum = warpReduceSum(total_sum);
      total_sum_sq = warpReduceSum(total_sum_sq);
      
      if (tid == 0) {
         accscalar_t mean = total_sum / normalized_size;
         accscalar_t var = total_sum_sq / normalized_size - mean * mean;
         accscalar_t inv_std = accscalar_t(1) / sqrt(var + eps);
         // Store mean and inv_std for use in normalization
         s_sum[0] = mean;
         s_sum_sq[0] = inv_std;
      }
  }
  __syncthreads();
  
  // Retrieve mean and inverse standard deviation
  accscalar_t mean = s_sum[0];
  accscalar_t inv_std = s_sum_sq[0];
  
  // Normalize the input and apply affine transformation
  for (int i = tid; i < normalized_size; i += blockDim.x) {
     accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
     accscalar_t norm_val = (val - mean) * inv_std;
     out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(weight[i]) + static_cast<accscalar_t>(bias[i]));
  }
}

// C++ interface for the optimized kernel
torch::Tensor layernorm_forward_optimized(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);

  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Determine threads: round up to a multiple of 32 and cap at 1024
  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  threads = ((threads + 31) / 32) * 32;
  if (threads > 1024) threads = 1024;
  int numWarps = threads / 32;
  int sharedMemSize = numWarps * 2 * sizeof(at::acc_type<float, true>);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_optimized_cuda", ([&] {
    layernorm_forward_kernel_optimized<scalar_t><<<outer_size, threads, sharedMemSize>>>(
       x.data_ptr<scalar_t>(),
       weight.data_ptr<scalar_t>(),
       bias.data_ptr<scalar_t>(),
       static_cast<float>(eps),
       output.data_ptr<scalar_t>(),
       normalized_size);
  }));

  return output;
}

// Binding to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &layernorm_forward_optimized, "LayerNorm forward optimized (CUDA)",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
