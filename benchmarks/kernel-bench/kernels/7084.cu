#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
  // Using 2D blocks for better mapping to outer x inner structure
  const int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int inner_idx = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (outer_idx >= outer || inner_idx >= inner) return;

  // Starting index for reduction in the r dimension
  const int base = outer_idx * (r * inner) + inner_idx;
  
  // Thread coarsening: each thread processes multiple elements at once
  constexpr int ITEMS_PER_THREAD = 4;
  scalar_t min_val = input[base];
  
  #pragma unroll
  for (int j = 0; j < r; j += ITEMS_PER_THREAD) {
    // Vector-style loading for better memory coalescing
    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD && (j + k) < r; k++) {
      const int index = base + (j + k) * inner;
      const scalar_t curr = input[index];
      min_val = curr < min_val ? curr : min_val;
    }
  }
  
  // Use shared memory for block-level reduction if needed
  __shared__ scalar_t shared_min[256]; // 16x16 thread block
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  shared_min[tid] = min_val;
  __syncthreads();
  
  // Additional reduction in shared memory if we have multiple items per thread
  if (tid < 128) {
    scalar_t other = shared_min[tid + 128];
    shared_min[tid] = min_val = (other < min_val) ? other : min_val;
  }
  __syncthreads();
  
  if (tid < 64) {
    scalar_t other = shared_min[tid + 64];
    shared_min[tid] = min_val = (other < min_val) ? other : min_val;
  }
  __syncthreads();
  
  // Final warp is fully active, no need for synchronization
  if (tid < 32) {
    volatile scalar_t* smem = shared_min;
    scalar_t other = smem[tid + 32];
    smem[tid] = min_val = (other < min_val) ? other : min_val;
    other = smem[tid + 16];
    smem[tid] = min_val = (other < min_val) ? other : min_val;
    other = smem[tid + 8];
    smem[tid] = min_val = (other < min_val) ? other : min_val;
    other = smem[tid + 4];
    smem[tid] = min_val = (other < min_val) ? other : min_val;
    other = smem[tid + 2];
    smem[tid] = min_val = (other < min_val) ? other : min_val;
    other = smem[tid + 1];
    smem[tid] = min_val = (other < min_val) ? other : min_val;
  }
  
  // Write result
  if (tid == 0) {
    output[outer_idx * inner + inner_idx] = shared_min[0];
  }
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  
  auto output = torch::empty(output_shape, input.options());

  // Use 2D thread blocks
  const dim3 threads(16, 16);
  const dim3 blocks(
    (outer + threads.x - 1) / threads.x,
    (inner + threads.y - 1) / threads.y
  );

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_cuda", ([&] {
    min_reduce_kernel<scalar_t><<<blocks, threads, 0, 
      c10::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        outer,
        r,
        inner);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction over a specified dimension (CUDA)");
}