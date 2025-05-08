#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// CUDA kernel that performs min reduction over a specified dimension with loop unrolling by factor 8
// The input is logically viewed as [outer, r, inner] and the reduction is performed over dimension r.

template <typename scalar_t>
__global__ void min_reduce_kernel(
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
  int base_offset = outer_idx * r * inner; // starting position for this reduction

  // Initialize min_val with the first element
  scalar_t min_val = input[base_offset + inner_idx];

  // Unroll the loop in chunks of 8 starting from j = 1
  int j_stop = ((r - 1) / 8) * 8 + 1;  // Process in multiples of 8
  #pragma unroll
  for (int j = 1; j < j_stop; j += 8) {
    int idx0 = base_offset + (j + 0) * inner + inner_idx;
    int idx1 = base_offset + (j + 1) * inner + inner_idx;
    int idx2 = base_offset + (j + 2) * inner + inner_idx;
    int idx3 = base_offset + (j + 3) * inner + inner_idx;
    int idx4 = base_offset + (j + 4) * inner + inner_idx;
    int idx5 = base_offset + (j + 5) * inner + inner_idx;
    int idx6 = base_offset + (j + 6) * inner + inner_idx;
    int idx7 = base_offset + (j + 7) * inner + inner_idx;

    min_val = min(min_val, input[idx0]);
    min_val = min(min_val, input[idx1]);
    min_val = min(min_val, input[idx2]);
    min_val = min(min_val, input[idx3]);
    min_val = min(min_val, input[idx4]);
    min_val = min(min_val, input[idx5]);
    min_val = min(min_val, input[idx6]);
    min_val = min(min_val, input[idx7]);
  }

  // Process any remaining elements
  for (int j = j_stop; j < r; j++) {
    int index = base_offset + j * inner + inner_idx;
    min_val = min(min_val, input[index]);
  }

  output[idx] = min_val;
}

// The forward function wraps the kernel launch and handles tensor shape rearrangement

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute the sizes for outer, reduction dimension (r), and inner dimensions
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Form the output shape by removing the reduced dimension
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
    min_reduce_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
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
