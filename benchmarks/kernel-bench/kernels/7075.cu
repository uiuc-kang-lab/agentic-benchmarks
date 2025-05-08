#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// This kernel maps the outer and inner dimensions directly to the 2D grid.
// Each block handles one output element corresponding to an (outer, inner) pair.
// Within each block, threads cooperatively reduce along the r dimension using shared memory.

template <typename scalar_t>
__global__ void min_reduce_2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int r,
    const int inner) {
  // Map the grid indices directly to the output coordinates
  int outer_idx = blockIdx.y;  // corresponds to the outer dimension
  int inner_idx = blockIdx.x;  // corresponds to the inner dimension

  // Compute the base pointer offset for this block's input slice
  // Input shape is logically [outer, r, inner].
  int base = outer_idx * (r * inner) + inner_idx;

  // Each block reduces over the r dimension for one output element
  int tid = threadIdx.x;
  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  // Each thread processes multiple elements of the r dimension in strides of blockDim.x
  for (int j = tid; j < r; j += blockDim.x) {
    // Index into the input: move j steps in the r dimension (stride = inner)
    int idx = base + j * inner;
    scalar_t val = input[idx];
    local_min = val < local_min ? val : local_min;
  }

  // Allocate shared memory for intra-block reduction
  extern __shared__ scalar_t sdata[];
  sdata[tid] = local_min;
  __syncthreads();

  // Perform reduction in shared memory
  // Perform tree-based reduction
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] = sdata[tid] < sdata[tid + s] ? sdata[tid] : sdata[tid + s];
    }
    __syncthreads();
  }

  // Unroll the last warp
  if (tid < 32) {
    volatile scalar_t *vsmem = sdata;
    vsmem[tid] = vsmem[tid] < vsmem[tid + 32] ? vsmem[tid] : vsmem[tid + 32];
    vsmem[tid] = vsmem[tid] < vsmem[tid + 16] ? vsmem[tid] : vsmem[tid + 16];
    vsmem[tid] = vsmem[tid] < vsmem[tid + 8] ? vsmem[tid] : vsmem[tid + 8];
    vsmem[tid] = vsmem[tid] < vsmem[tid + 4] ? vsmem[tid] : vsmem[tid + 4];
    vsmem[tid] = vsmem[tid] < vsmem[tid + 2] ? vsmem[tid] : vsmem[tid + 2];
    vsmem[tid] = vsmem[tid] < vsmem[tid + 1] ? vsmem[tid] : vsmem[tid + 1];
  }

  // Thread 0 writes the result for this block to the output
  if (tid == 0) {
    // Output is shaped as [outer, inner], stored in row-major order
    output[outer_idx * inner + inner_idx] = sdata[0];
  }
}

// Forward function: computes sizes and launches the kernel using 2D grid indexing
// The reduction dimension is determined by the input dim parameter

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute sizes: outer = product of dimensions before 'dim', r is size at 'dim', inner = product of dimensions after 'dim'
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Build output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Map the output to a 2D grid:
  //   gridDim.x corresponds to the inner dimension
  //   gridDim.y corresponds to the outer dimension
  dim3 blocks(inner, outer);
  const int threads = 128;  // A choice that can be tuned
  const size_t shared_mem_size = threads * sizeof(float); // Note: This assumes float; AT_DISPATCH_ALL_TYPES will adapt size

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_2d_kernel_cuda", ([&] {
    min_reduce_2d_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t),
      c10::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        r,
        inner);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction over a specified dimension using 2D indexing (CUDA)");
}
