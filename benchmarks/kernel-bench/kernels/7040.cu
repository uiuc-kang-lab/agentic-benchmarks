#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t reduce_min_along_dim(
    const scalar_t* __restrict__ input,
    const int base,
    const int r,
    const int inner,
    const int outer_idx,
    const int inner_idx) {
  scalar_t min_val = input[base];
  #pragma unroll
  for (int j = 1; j < r; j++) {
    int index = outer_idx * (r * inner) + j * inner + inner_idx;
    scalar_t curr = input[index];
    min_val = curr < min_val ? curr : min_val;
  }
  return min_val;
}

template <typename scalar_t>
__global__ void min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = outer * inner;
  if (idx >= total) return;

  const int outer_idx = idx / inner;
  const int inner_idx = idx % inner;
  const int base = outer_idx * (r * inner) + inner_idx;

  output[idx] = reduce_min_along_dim(
    input, base, r, inner, outer_idx, inner_idx);
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

  const int total = outer * inner;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_cuda", ([&] {
    min_reduce_kernel<scalar_t><<<blocks, threads, 0, 
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