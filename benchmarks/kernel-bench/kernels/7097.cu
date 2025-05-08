#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// Using shared memory for each block to perform the reduction
// This reduces global memory access and improves performance.
template <typename scalar_t>
__global__ void min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
  extern __shared__ scalar_t shared_min[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (idx >= total) return;

  int outer_idx = idx / inner;
  int inner_idx = idx % inner;
  int base = outer_idx * (r * inner) + inner_idx;
  
  scalar_t min_val = input[base];

  // Load elements into shared memory
  for (int j = 0; j < r; j++) {
    int index = base + j * inner;
    scalar_t curr = input[index];
    min_val = min(min_val, curr);
  }
  shared_min[threadIdx.x] = min_val;
  __syncthreads();

  // Reduce within shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared_min[threadIdx.x] = min(shared_min[threadIdx.x], shared_min[threadIdx.x + s]);
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (threadIdx.x == 0) {
    output[blockIdx.x] = shared_min[0];
  }
}

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

  int total = outer * inner;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_cuda", ([&] {
    min_reduce_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t), 
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