#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// This kernel uses warp-level primitives along with shared memory to reduce global memory latency.
// Each block handles one output element. Threads in a warp compute a local minimum using __shfl_down_sync, 
// then the warp leaders store their results in shared memory. A final reduction is applied to these values in shared memory.

template <typename scalar_t>
__global__ void min_reduce_shared_warp_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {

  // Each block computes the reduction for one output element
  int idx = blockIdx.x; // idx in [0, outer * inner)
  int outer_idx = idx / inner;
  int inner_idx = idx % inner;
  int base = outer_idx * (r * inner) + inner_idx;

  int tid = threadIdx.x;
  // Initialize to maximum value
  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  // Strided load along the reduction dimension
  for (int i = tid; i < r; i += blockDim.x) {
    int pos = base + i * inner;
    scalar_t val = input[pos];
    local_min = (val < local_min) ? val : local_min;
  }

  // Perform warp-level reduction using shuffle intrinsics
  unsigned int full_mask = 0xffffffff;
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    scalar_t temp = __shfl_down_sync(full_mask, local_min, offset);
    local_min = (temp < local_min) ? temp : local_min;
  }

  // Allocate shared memory to store per-warp results
  extern __shared__ char smem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

  // Each warp's lane 0 writes its result to shared memory
  int warp_id = tid / warpSize;
  if ((tid & (warpSize - 1)) == 0) {
    sdata[warp_id] = local_min;
  }
  __syncthreads();

  // Final reduction: let the first warp reduce the results from each warp
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  if (tid < num_warps) {
    local_min = sdata[tid];
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
      scalar_t temp = __shfl_down_sync(full_mask, local_min, offset);
      local_min = (temp < local_min) ? temp : local_min;
    }
    // Write the final result back to shared memory
    if (tid == 0) {
      sdata[0] = local_min;
    }
  }
  __syncthreads();

  // Thread 0 writes the final minimum to the output tensor
  if (tid == 0) {
    output[idx] = sdata[0];
  }
}

// Host function to launch the kernel
torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }
  
  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate outer, reduction (r), and inner sizes
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Create output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Launch one block per output element
  int blocks = outer * inner;
  int threads = 256; // Using 256 threads per block
  // Calculate shared memory size: one scalar_t per warp
  int num_warps = (threads + 31) / 32;
  size_t shared_mem_size = num_warps * sizeof(scalar_t);

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_shared_warp_cuda", ([&] {
    min_reduce_shared_warp_kernel<scalar_t><<<blocks, threads, shared_mem_size, 
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
  m.def("forward", &forward, "Min reduction over a specified dimension using shared memory warp reduction (CUDA)");
}
