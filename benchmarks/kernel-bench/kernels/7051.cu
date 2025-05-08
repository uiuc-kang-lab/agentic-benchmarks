#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Optimized CUDA kernel combining warp-level and 2D grid strategies for better performance.
template <typename scalar_t>
__global__ void optimized_min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
  // Use combined 2D grid and warp-level reduction for efficiency.
  int outer_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (outer_idx >= outer || inner_idx >= inner) return;

  int lane = threadIdx.x % 32;
  int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (warpId >= outer * inner) return;

  int base = outer_idx * (r * inner) + inner_idx;
  scalar_t my_min = std::numeric_limits<scalar_t>::max();
  for (int j = lane; j < r; j += 32) {
    int pos = base + j * inner;
    scalar_t val = input[pos];
    if (val < my_min) {
      my_min = val;
    }
  }

  // Warp-level reduction using __shfl_down_sync
  for (int offset = 16; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, my_min, offset);
    if (other < my_min) {
      my_min = other;
    }
  }

  if (lane == 0) {
    atomicMin(&output[outer_idx * inner + inner_idx], my_min);
  }
}

// Forward function prepares the tensor dimensions and launches the optimized kernel
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

  const dim3 threads(32, 16);
  const dim3 blocks((inner + threads.x - 1) / threads.x, (outer + threads.y - 1) / threads.y);

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "optimized_min_reduce_cuda", ([&] {
    optimized_min_reduce_kernel<scalar_t><<<blocks, threads, 0, 
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
  m.def("forward", &forward, "Optimized min reduction over a specified dimension using combined strategies (CUDA)");
}