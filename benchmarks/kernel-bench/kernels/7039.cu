#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// CUDA warp-level min reduction function
__inline__ __device__ float warpMin(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = min(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// CUDA kernel for performing a min reduction along a specified dimension using warp-level primitives.
template <typename scalar_t>
__global__ void min_reduce_kernel_warp(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (idx >= total) return;

  int outer_idx = idx / inner;
  int inner_idx = idx % inner;

  // Starting index for reduction in the r dimension.
  int base = outer_idx * (r * inner) + inner_idx;
  scalar_t min_val = input[base];

  for (int j = 1; j < r; j++) {
    int index = outer_idx * (r * inner) + j * inner + inner_idx;
    scalar_t curr = input[index];
    min_val = min(min_val, curr);
  }

  // Use warp-level primitive to determine minimum value
  min_val = warpMin(min_val);

  // Output the first thread's result in the warp
  if (threadIdx.x % warpSize == 0) {
    output[outer_idx * inner + inner_idx] = min_val;
  }
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate sizes: 'outer' are dimensions before reduced dimension,
  // 'r' is the size of the reduced dimension, and 'inner' are dimensions after.
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

  int total = outer * inner;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_cuda_warp", ([&] {
    min_reduce_kernel_warp<scalar_t><<<blocks, threads, 0, 
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
  m.def("forward", &forward, "Min reduction over a specified dimension (CUDA)");
}