#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Modular device function to perform warp-level minimum reduction using shuffle operations
template <typename scalar_t>
__device__ inline scalar_t warpReduceMin(scalar_t val) {
  // Full mask for all lanes
  for (int offset = 16; offset > 0; offset /= 2) {
    scalar_t temp = __shfl_down_sync(0xffffffff, val, offset);
    val = (temp < val) ? temp : val;
  }
  return val;
}

// Modular device function to perform block-level reduction using shared memory
template <typename scalar_t>
__device__ inline scalar_t blockReduceMin(scalar_t val) {
  int lane = threadIdx.x & 31;       // lane index within the warp
  int wid  = threadIdx.x >> 5;        // warp index within the block
  
  // First, reduce within each warp
  val = warpReduceMin<scalar_t>(val);

  // Allocate shared memory for storing each warp's result
  __shared__ scalar_t shared[32];
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  // Only the first warp loads the results from shared memory
  int numWarps = (blockDim.x + 31) >> 5;
  val = (threadIdx.x < numWarps) ? shared[lane] : std::numeric_limits<scalar_t>::max();
  if (wid == 0) {
    val = warpReduceMin<scalar_t>(val);
  }
  return val;
}

// Kernel for modular min reduction over a specified dimension
// The input tensor is logically reshaped into [outer, r, inner], and reduction is over the r dimension

template <typename scalar_t>
__global__ void modular_min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
  // Each block computes one output element corresponding to a unique (outer, inner) pair
  int idx = blockIdx.x;
  if (idx >= outer * inner) return;

  int outer_idx = idx / inner;
  int inner_idx = idx % inner;
  
  // Compute pointer to the beginning of the reduction segment
  const scalar_t* in_ptr = input + outer_idx * (r * inner) + inner_idx;
  
  // Each thread processes a chunk of the r dimension using striding
  scalar_t local_min = std::numeric_limits<scalar_t>::max();
  for (int j = threadIdx.x; j < r; j += blockDim.x) {
    scalar_t val = in_ptr[j * inner];
    local_min = (val < local_min) ? val : local_min;
  }
  
  // Use modular block reduction to combine the partial results
  local_min = blockReduceMin<scalar_t>(local_min);
  
  // Thread 0 writes the final minimum value to the output
  if (threadIdx.x == 0) {
    output[idx] = local_min;
  }
}

// Host forward function
// Reshapes input into dimensions [outer, r, inner] according to the reduction dimension and launches the kernel

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate sizes: 'outer' are dimensions before dim, 'r' is the reduction dimension, 'inner' are dimensions after dim
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

  // Allocate output tensor
  auto output = torch::empty(output_shape, input.options());

  // Compute total number of reduction groups
  int total = outer * inner;

  // Configure kernel launch parameters
  int threads = 256; // Using 256 threads per block
  int blocks = total; // One block per reduction group

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "modular_min_reduce_cuda", ([&] {
    int shared_mem_size = 0;
    modular_min_reduction_kernel<scalar_t><<<blocks, threads, shared_mem_size, c10::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        outer,
        r,
        inner);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Modular CUDA kernel min reduction over a specified dimension");
}
