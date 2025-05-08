#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// This kernel uses shared memory for intra-block reduction and
// warp-level primitives (__shfl_down_sync) for the final stage.
// Each block computes the min reduction along the r-dimension for one output element.

template <typename scalar_t>
__global__ void min_reduce_shared_warp_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {
  // Map block index to output element
  int out_idx = blockIdx.x;
  int outer_idx = out_idx / inner;
  int inner_idx = out_idx % inner;
  int base = outer_idx * (r * inner) + inner_idx;

  // Each thread computes a partial minimum over the reduction dimension
  scalar_t local_min = std::numeric_limits<scalar_t>::max();
  for (int j = threadIdx.x; j < r; j += blockDim.x) {
    int idx = base + j * inner;
    scalar_t val = input[idx];
    if (val < local_min) {
      local_min = val;
    }
  }

  // Allocate shared memory dynamically
  extern __shared__ char smem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
  sdata[threadIdx.x] = local_min;
  __syncthreads();

  // Intra-block reduction using shared memory
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s) {
      scalar_t other = sdata[threadIdx.x + s];
      sdata[threadIdx.x] = sdata[threadIdx.x] < other ? sdata[threadIdx.x] : other;
    }
    __syncthreads();
  }

  // Final warp-level reduction using __shfl_down_sync
  if (threadIdx.x < 32) {
    scalar_t val = sdata[threadIdx.x];
    // Unroll the warp reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
      scalar_t shfl_val = __shfl_down_sync(0xffffffff, val, offset);
      val = val < shfl_val ? val : shfl_val;
    }
    if (threadIdx.x == 0) {
      output[out_idx] = val;
    }
  }
}

// Forward function: sets up tensor dimensions, output shape, and kernel launch parameters.

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate dimensions: outer (product of dims before the reduced dim),
  // r (size of the reduced dimension), inner (product of dims after the reduced dim)
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

  // Each block computes one output element
  int num_blocks = outer * inner;
  const int threads = 256;
  size_t shared_mem_size = threads * input.element_size();

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_shared_warp_cuda", ([&] {
    min_reduce_shared_warp_kernel<scalar_t><<<num_blocks, threads, shared_mem_size,
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
  m.def("forward", &forward, "Min reduction over a specified dimension using shared memory and warp-level primitives (CUDA)");
}
