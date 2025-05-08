#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// Optimized CUDA kernel for performing a min reduction along a specified dimension
// utilizing shared memory for reduction.
template <typename scalar_t>
__global__ void min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
  extern __shared__ scalar_t sdata[];
  // Each block computes one output element corresponding to an (outer, inner) pair
  int out_idx = blockIdx.x;
  int outer_idx = out_idx / inner;
  int inner_idx = out_idx % inner;
  int input_base = outer_idx * (r * inner) + inner_idx;

  // Each thread computes a partial minimum over a slice of the reduction dimension
  // Initialize with a high value; assuming floating-point, INFINITY works
  // For integer types one might use a very high value (or specialize the kernel)
  scalar_t thread_min = INFINITY;
  for (int j = threadIdx.x; j < r; j += blockDim.x) {
    scalar_t curr = input[input_base + j * inner];
    thread_min = min(thread_min, curr);
  }

  sdata[threadIdx.x] = thread_min;
  __syncthreads();

  // Parallel reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] = min(sdata[threadIdx.x], sdata[threadIdx.x + s]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    output[out_idx] = sdata[0];
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

  size_t shared_mem_size = threads * sizeof(scalar_t);

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_cuda_optimized", ([&] {
    min_reduce_kernel_optimized<scalar_t><<<blocks, threads, shared_mem_size, 
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
  m.def("forward", &forward, "Optimized Min reduction over a specified dimension (CUDA)");
}
