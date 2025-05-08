#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void min_reduce_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int outer, int r, int inner) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= outer * inner) return;
  
  int outer_idx = idx / inner;
  int inner_idx = idx % inner;
  int base = outer_idx * r * inner + inner_idx;
  
  scalar_t min_val = input[base];
  for (int j = 1; j < r; ++j) {
    scalar_t curr = input[base + j*inner];
    min_val = (curr < min_val) ? curr : min_val;
  }
  output[idx] = min_val;
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  input = input.contiguous();
  
  int ndim = input.dim();
  int outer = 1;
  for (int i = 0; i < dim; ++i) outer *= input.size(i);
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim+1; i < ndim; ++i) inner *= input.size(i);

  auto sizes = input.sizes().vec();
  sizes.erase(sizes.begin() + dim);
  auto output = torch::empty(sizes, input.options());
  
  const int threads = 256;
  dim3 blocks((outer*inner + threads-1)/threads);
  
  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce", ([&] {
    min_reduce_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), outer, r, inner);
  }));
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized min reduction");
}