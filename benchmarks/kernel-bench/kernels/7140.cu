#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Warp-level reduction using shuffle instructions in a branch-free manner.
// This function reduces a value across a warp without divergent branching.
template <typename scalar_t>
__inline__ __device__ scalar_t warpMin(scalar_t val) {
  // Use full mask
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(__activemask(), val, offset);
    // Use branch-free min operation
    val = (other < val) ? other : val;
  }
  return val;
}

// Kernel performing min reduction with minimized warp divergence by using warp shuffle
// to perform branch-free reductions within warps and then across warps using shared memory.
// The input is logically reshaped as [outer, r, inner] and the reduction is performed over the r dimension.

template <typename scalar_t>
__global__ void no_divergence_min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  // Each block processes one output element
  int idx = blockIdx.x;
  if (idx >= outer * inner) return;

  int outer_idx = idx / inner;
  int inner_idx = idx % inner;

  // Pointer to the reduction group in input for this output element
  const scalar_t* in_ptr = input + outer_idx * (r * inner) + inner_idx;

  // Each thread computes a partial minimum over its assigned elements in the r dimension
  scalar_t local_min = std::numeric_limits<scalar_t>::max();
  for (int j = threadIdx.x; j < r; j += blockDim.x) {
    // Compute index and get the value
    scalar_t curr = in_ptr[j * inner];
    // Use branch-free min update
    local_min = (curr < local_min) ? curr : local_min;
  }

  // Perform warp-level reduction using shuffle instructions
  local_min = warpMin(local_min);

  // Each warp's first thread writes its result to shared memory
  unsigned int lane = threadIdx.x & (warpSize - 1);
  __shared__ scalar_t shared[32];  // supports up to 1024 threads per block (32 warps)
  if (lane == 0) {
    shared[threadIdx.x / warpSize] = local_min;
  }
  __syncthreads();

  // Let the first warp load the partial results from shared memory
  int numWarps = blockDim.x / warpSize;
  scalar_t block_min = (threadIdx.x < numWarps) ? shared[threadIdx.x] : std::numeric_limits<scalar_t>::max();
  
  // Final reduction within the first warp
  block_min = warpMin(block_min);

  // The first thread writes the final result for this block
  if (threadIdx.x == 0) {
    output[idx] = block_min;
  }
}

// Host forward function for performing min reduction along a specified dimension.
// The input tensor is logically reshaped as [outer, r, inner] where r is the reduction dim.
// The output tensor has the reduced dimension removed.

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute sizes: outer dimensions before 'dim', 'r' is the size along 'dim', inner dimensions after.
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Construct output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }

  // Allocate output tensor
  auto output = torch::empty(output_shape, input.options());

  // Each block handles one output element (i.e. one reduction group)
  int total = outer * inner;

  // Use 256 threads per block
  int threads = 256;
  int blocks = total;  // One block per reduction group

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "no_divergence_min_reduce_cuda", ([&] {
    int shared_mem_size = 32 * sizeof(scalar_t);  // Shared memory for warp leaders
    no_divergence_min_reduction_kernel<scalar_t><<<blocks, threads, shared_mem_size, 
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
  m.def("forward", &forward, "Min reduction over a specified dimension with minimized warp divergence (CUDA)");
}
