#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void min_reduce_kernel_2d(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
  
  int outer_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
  
  if (outer_idx >= outer || inner_idx >= inner) return;

  int base = outer_idx * (r * inner) + inner_idx;
  scalar_t min_val = input[base];

  for (int j = 1; j < r; j++) {
    int index = outer_idx * (r * inner) + j * inner + inner_idx;
    scalar_t curr = input[index];
    if (curr < min_val) {
      min_val = curr;
    }
  }
  output[outer_idx * inner + inner_idx] = min_val;
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

  dim3 threadsPerBlock(32, 8);  // 256 threads per block
  dim3 numBlocks(
    (outer + threadsPerBlock.y - 1) / threadsPerBlock.y,
    (inner + threadsPerBlock.x - 1) / threadsPerBlock.x
  );

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_cuda_2d", ([&] {
    min_reduce_kernel_2d<scalar_t><<<numBlocks, threadsPerBlock, 0, 
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
  m.def("forward", &forward, "Optimized 2D min reduction (CUDA)");
}
