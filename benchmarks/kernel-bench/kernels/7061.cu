#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Optimized CUDA kernel using 2D grid indexing for min reduction.
// The input is logically reshaped as [outer, r, inner] and the reduction is performed along the r dimension.
// Each warp computes the min reduction for one output element.

template <typename scalar_t>
__global__ void min_reduce_2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  const int warpSize = 32;
  // Map the outer dimension to gridDim.y
  int outer_idx = blockIdx.y;

  // Each block is organized so that several warps process different elements along the inner dimension.
  int warpsPerBlock = blockDim.x / warpSize;  // e.g. 4 warps per block using 128 threads
  int warp_id_in_block = threadIdx.x / warpSize;
  // Compute global inner index across blocks
  int inner_idx = blockIdx.x * warpsPerBlock + warp_id_in_block;
  if(inner_idx >= inner) return;

  // Base pointer into the reduction axis
  int base = outer_idx * (r * inner) + inner_idx;
  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  // Each thread in the warp processes a portion of the reduction dimension with stride warpSize
  int lane = threadIdx.x % warpSize;
  #pragma unroll
  for (int j = lane; j < r; j += warpSize) {
    int idx = base + j * inner;
    scalar_t val = input[idx];
    if (val < local_min) {
      local_min = val;
    }
  }

  // Intra-warp reduction using __shfl_down_sync
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, local_min, offset);
    if (other < local_min) {
      local_min = other;
    }
  }

  // The first lane of each warp writes the result
  if (lane == 0) {
    output[outer_idx * inner + inner_idx] = local_min;
  }
}

// Forward function: prepares tensor dimensions and launches the kernel

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute dimensions:
  // outer: product of dimensions before the reduction dimension
  // r: size of the reduction dimension
  // inner: product of dimensions after the reduction dimension
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

  // Use 2D grid indexing to map outputs directly to thread blocks:
  // gridDim.y corresponds to 'outer' and gridDim.x to 'inner' (distributed over warps per block)
  const int warpsPerBlock = 4;                  // Tunable: number of warps per block
  const int blockSize = warpsPerBlock * warpSize; // e.g., 4*32 = 128 threads per block
  int gridDimX = (inner + warpsPerBlock - 1) / warpsPerBlock;
  dim3 gridDim(gridDimX, outer);

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_2d_cuda", ([&] {
    min_reduce_2d_kernel<scalar_t><<<gridDim, blockSize, 0,
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
  m.def("forward", &forward, "Min reduction over a specified dimension using 2D grid indexing (CUDA)");
}
