#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Constant memory for read-only data
__constant__ int const_outer;
__constant__ int const_inner;
__constant__ int const_r;

// Each warp computes the min reduction for one output element using warp-level primitives
// The input is logically reshaped as [outer, r, inner], and reduction is performed along the r dimension.
template <typename scalar_t>
__global__ void min_reduce_warp_const_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output) {
  // Compute global warp id: each warp is responsible for one output element
  int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (warpId >= const_outer * const_inner) return;

  int outer_idx = warpId / const_inner;
  int inner_idx = warpId % const_inner;
  int base = outer_idx * (const_r * const_inner) + inner_idx;

  // Compute lane id within the warp
  int lane = threadIdx.x % 32;

  // Each thread computes a partial min over the reduction dimension with stride = warpSize
  scalar_t my_min = std::numeric_limits<scalar_t>::max();
  for (int j = lane; j < const_r; j += 32) {
    int pos = base + j * const_inner;
    scalar_t val = input[pos];
    if (val < my_min) {
      my_min = val;
    }
  }

  // Warp-level reduction using __shfl_down_sync to combine the results
  for (int offset = 16; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, my_min, offset);
    if (other < my_min) {
      my_min = other;
    }
  }

  // The first lane of the warp writes the result
  if (lane == 0) {
    output[warpId] = my_min;
  }
}

// Host function to set up the kernel launch
torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate sizes: outer dimensions, size of reduction dimension (r), and inner dimensions
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Create the output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Copy constants to constant memory
  cudaMemcpyToSymbol(const_outer, &outer, sizeof(int));
  cudaMemcpyToSymbol(const_inner, &inner, sizeof(int));
  cudaMemcpyToSymbol(const_r, &r, sizeof(int));

  // Each output element is processed by one warp (32 threads)
  int total_warps = outer * inner;
  int threads_per_block = 128;  // 128 threads per block gives 4 warps per block
  int num_blocks = (total_warps * 32 + threads_per_block - 1) / threads_per_block;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_warp_const_cuda", ([&] {
    min_reduce_warp_const_kernel<scalar_t><<<num_blocks, threads_per_block, 0,
      c10::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>());
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction over a specified dimension using constant memory (CUDA)");
}