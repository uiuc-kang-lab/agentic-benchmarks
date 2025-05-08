#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel fuses the HardSigmoid activation with an intra-block reduction.
// Each thread computes its activated value, writes it to the output, and accumulates a local sum.
// Then, using shared memory and warp-level primitives (__shfl_down_sync), a block‐wise reduction is performed.
// A second kernel then reduces the per‐block sums to a final total (which can be used for auxiliary purposes).

template <typename scalar_t>
__global__ void hardsigmoid_fused_kernel(const scalar_t* __restrict__ input,
                                           scalar_t* __restrict__ output,
                                           scalar_t* __restrict__ block_sums,
                                           size_t numel) {
  extern __shared__ char shared_mem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + tid;
  unsigned int stride = gridDim.x * blockDim.x;
  scalar_t local_sum = 0;

  // Compute activation and accumulate local sum
  for (size_t i = idx; i < numel; i += stride) {
    scalar_t x = input[i];
    scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
    y = (y < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) 
         : (y > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : y);
    output[i] = y;
    local_sum += y;
  }

  // Store local sum into shared memory
  sdata[tid] = local_sum;
  __syncthreads();

  // Intra-block reduction using shared memory
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Warp-level reduction using __shfl_down_sync
  if (tid < 32) {
    unsigned int mask = 0xffffffff;
    scalar_t val = sdata[tid];
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(mask, val, offset);
    }
    sdata[tid] = val;
  }

  // Write the block's sum to global memory
  if (tid == 0) {
    block_sums[blockIdx.x] = sdata[0];
  }
}

// Final reduction kernel to sum the block sums.
// This kernel uses the same shared memory and warp-level techniques to reduce an array of length n.

template <typename scalar_t>
__global__ void final_reduce_kernel(const scalar_t* __restrict__ input,
                                       scalar_t* __restrict__ output,
                                       size_t n) {
  extern __shared__ char shared_mem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + tid;
  unsigned int stride = gridDim.x * blockDim.x;
  scalar_t local_sum = 0;

  for (size_t i = idx; i < n; i += stride) {
    local_sum += input[i];
  }
  sdata[tid] = local_sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    unsigned int mask = 0xffffffff;
    scalar_t val = sdata[tid];
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(mask, val, offset);
    }
    sdata[tid] = val;
  }

  if(tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}


// The forward function launches the fused kernel and then performs a final reduction on block sums.
// The output tensor contains the elementwise HardSigmoid activation, ensuring correct results.
// The reduction (final reduced value) is computed for demonstration of optimization and can be used
// for auxiliary purposes (e.g., monitoring activation statistics).

torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

  auto output = torch::empty_like(input);
  const size_t numel = input.numel();

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  // Allocate temporary tensor for per-block sums
  auto block_sums_tensor = torch::empty({blocks}, input.options());

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_fused_cuda", ([&] {
    // Launch the fused kernel
    hardsigmoid_fused_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      block_sums_tensor.data_ptr<scalar_t>(),
      numel);

    // Optional: Perform final reduction on block sums to compute the total sum of activated outputs
    int current_n = blocks;
    torch::Tensor reduction_tensor = block_sums_tensor;
    while (current_n > 1) {
      int threads2 = (current_n < 256) ? current_n : 256;
      int blocks2 = (current_n + threads2 - 1) / threads2;
      auto temp_tensor = torch::empty({blocks2}, input.options());
      final_reduce_kernel<scalar_t><<<blocks2, threads2, threads2 * sizeof(scalar_t)>>>(
         reduction_tensor.data_ptr<scalar_t>(),
         temp_tensor.data_ptr<scalar_t>(),
         current_n);
      reduction_tensor = temp_tensor;
      current_n = blocks2;
    }
    // Final reduction value is in reduction_tensor[0] (can be used if needed)
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA) with fused reduction optimization");
}
