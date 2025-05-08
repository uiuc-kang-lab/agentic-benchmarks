#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Define constants for 2D block configuration
#define TILE_SIZE 8                // Number of output elements processed per block
#define THREADS_PER_REDUCTION 128  // Number of threads used for reducing over the 'r' dimension

// Kernel: Each block processes TILE_SIZE output elements concurrently.
// Block dimensions: (THREADS_PER_REDUCTION, TILE_SIZE).
// Each thread in a (tx, ty) processes a portion of r for one output element.

template <typename scalar_t>
__global__ void balanced_min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner,
    const int total) {
  // Determine which output element (within the global flattening of outer*inner) this thread tile is processing
  int tile_idx = threadIdx.y;  // index within the tile
  int global_out_idx = blockIdx.x * TILE_SIZE + tile_idx;  // overall output index
  
  // If global output index exceeds total outputs, exit
  if (global_out_idx >= total) return;
  
  // Compute outer and inner indices from global output index
  int outer_idx = global_out_idx / inner;
  int inner_idx = global_out_idx % inner;
  
  // Pointer to the beginning of the reduction segment in the input tensor
  // Input is logically [outer, r, inner] so element at [outer_idx, j, inner_idx] is at offset outer_idx*(r*inner) + j*inner + inner_idx
  const scalar_t* in_ptr = input + outer_idx * (r * inner) + inner_idx;

  // Each thread (along x direction) computes a partial minimum for this output element
  scalar_t local_min = std::numeric_limits<scalar_t>::max();
  for (int j = threadIdx.x; j < r; j += blockDim.x) {
    scalar_t val = in_ptr[j * inner];
    local_min = (val < local_min) ? val : local_min;
  }

  // Allocate shared memory for reduction; the shared memory array has TILE_SIZE rows and THREADS_PER_REDUCTION columns
  extern __shared__ char shared_mem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);

  // Each thread writes its partial result into shared memory
  int offset = threadIdx.y * blockDim.x + threadIdx.x;
  sdata[offset] = local_min;
  __syncthreads();

  // Perform reduction along the x dimension for the current tile row
  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      int index = threadIdx.y * blockDim.x + threadIdx.x;
      int index_plus = index + s;
      scalar_t a = sdata[index];
      scalar_t b = sdata[index_plus];
      sdata[index] = (b < a) ? b : a;
    }
    __syncthreads();
  }

  // The first thread in each tile row writes the computed minimum to the output tensor
  if (threadIdx.x == 0) {
    output[global_out_idx] = sdata[threadIdx.y * blockDim.x];
  }
}

// Host forward function
// Reshapes the input tensor to [outer, r, inner] based on 'dim', and reduces over the 'r' dimension

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute sizes: 'outer' are dimensions before 'dim', 'r' is size along 'dim', and 'inner' are dimensions after
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
  
  // Total number of output elements (i.e. reduction groups) in flattened [outer, inner] space
  int total = outer * inner;

  // Grid configuration: Each block processes TILE_SIZE output elements
  int blocks = (total + TILE_SIZE - 1) / TILE_SIZE;
  dim3 grid(blocks);
  dim3 block(THREADS_PER_REDUCTION, TILE_SIZE);

  // Shared memory size: TILE_SIZE * THREADS_PER_REDUCTION elements of type scalar_t
  int shared_mem_size = TILE_SIZE * THREADS_PER_REDUCTION * sizeof(float);
  // Use AT_DISPATCH_ALL_TYPES to handle various data types
  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "balanced_min_reduce_cuda", ([&] {
    balanced_min_reduction_kernel<scalar_t><<<grid, block, TILE_SIZE * THREADS_PER_REDUCTION * sizeof(scalar_t),
      c10::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        outer,
        r,
        inner,
        total);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Balanced min reduction over a specified dimension (CUDA)");
}
