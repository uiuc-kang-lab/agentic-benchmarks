#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Parallel min reduction kernel using shared memory and warp-level unrolling.
// Each CUDA block processes one reduction group (one output element) corresponding
// to the [outer, inner] index. Threads within the block compute partial minima over
// the reduction dimension and then collaborate via shared memory reduction.

template <typename scalar_t>
__global__ void fast_min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  // Each block processes one output element.
  int idx = blockIdx.x;
  if (idx >= outer * inner) return;

  int outer_idx = idx / inner;
  int inner_idx = idx % inner;

  // Pointer to the beginning of the reduction group in input
  const scalar_t* in_ptr = input + outer_idx * (r * inner) + inner_idx;

  // Each thread computes a partial minimum over its chunk of the r dimension.
  scalar_t local_min = std::numeric_limits<scalar_t>::max();
  for (int j = threadIdx.x; j < r; j += blockDim.x) {
    scalar_t val = in_ptr[j * inner];
    if (val < local_min)
      local_min = val;
  }

  // Allocate shared memory for reduction
  extern __shared__ char shared_mem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);
  sdata[threadIdx.x] = local_min;
  __syncthreads();

  // Perform reduction in shared memory using iterative halving
  // Unroll the last warp to avoid unnecessary synchronizations
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s) {
      scalar_t other = sdata[threadIdx.x + s];
      if (other < sdata[threadIdx.x])
        sdata[threadIdx.x] = other;
    }
    __syncthreads();
  }
  
  if (threadIdx.x < 32) {
    // Warp-level reduction without __syncthreads();
    volatile scalar_t* vsdata = sdata;
    vsdata[threadIdx.x] = (vsdata[threadIdx.x + 32] < vsdata[threadIdx.x]) ? vsdata[threadIdx.x + 32] : vsdata[threadIdx.x];
    vsdata[threadIdx.x] = (vsdata[threadIdx.x + 16] < vsdata[threadIdx.x]) ? vsdata[threadIdx.x + 16] : vsdata[threadIdx.x];
    vsdata[threadIdx.x] = (vsdata[threadIdx.x + 8]  < vsdata[threadIdx.x]) ? vsdata[threadIdx.x + 8]  : vsdata[threadIdx.x];
    vsdata[threadIdx.x] = (vsdata[threadIdx.x + 4]  < vsdata[threadIdx.x]) ? vsdata[threadIdx.x + 4]  : vsdata[threadIdx.x];
    vsdata[threadIdx.x] = (vsdata[threadIdx.x + 2]  < vsdata[threadIdx.x]) ? vsdata[threadIdx.x + 2]  : vsdata[threadIdx.x];
    vsdata[threadIdx.x] = (vsdata[threadIdx.x + 1]  < vsdata[threadIdx.x]) ? vsdata[threadIdx.x + 1]  : vsdata[threadIdx.x];
  }

  // Write the result from the first thread of the block
  if (threadIdx.x == 0) {
    output[idx] = sdata[0];
  }
}


// Host forward function for fast min reduction over a specified dimension.
// The input tensor is logically reshaped to [outer, r, inner] and the reduction
// is performed along the r dimension.
torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute sizes: outer dimensions before 'dim', 'r' is size along 'dim', and inner dimensions after 'dim'.
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Create output shape by removing the reduced dimension.
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  
  // Allocate output tensor.
  auto output = torch::empty(output_shape, input.options());

  // Total number of output elements (each corresponds to one reduction group).
  int total = outer * inner;

  // Determine block size. Using 256 threads per block is typical; if r is smaller, threads that do no work will
  // simply output the identity value.
  // Choose thread count based on device properties to maximize occupancy
int threads = 256; // Default to 256 threads per block for better occupancy
#ifdef __CUDA_ARCH__
if (r < 256) threads = (r + 31) & ~31; // Round up to nearest multiple of warp size
#endif
  int blocks = total;  // one block per reduction group

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "fast_min_reduce_cuda", ([&] {
    int shared_mem_size = threads * sizeof(scalar_t);
    fast_min_reduction_kernel<scalar_t><<<blocks, threads, shared_mem_size, c10::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        outer,
        r,
        inner);
  }));

  return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized block size min reduction over a specified dimension (CUDA)");
}
