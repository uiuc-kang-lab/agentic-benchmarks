#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Kernel using streams for overlapped memory operations and computation
template <typename scalar_t>
__global__ void min_reduce_streams_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {

  const int warpSize = 32;
  // Each warp calculates one output element
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (warp_id >= outer * inner) return;

  int lane = threadIdx.x % warpSize;
  int outer_idx = warp_id / inner;
  int inner_idx = warp_id % inner;
  int base = outer_idx * (r * inner) + inner_idx;

  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  #pragma unroll
  for (int j = lane; j < r; j += warpSize) {
    int idx = base + j * inner;
    scalar_t val = input[idx];
    local_min = (val < local_min) ? val : local_min;
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, local_min, offset);
    local_min = (other < local_min) ? other : local_min;
  }

  if (lane == 0) {
    output[warp_id] = local_min;
  }
}

// Forward function with memory transfer and kernel execution overlap
torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
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

  int total_output = outer * inner;
  const int threads_per_block = 128;
  int num_blocks = (total_output * 32 + threads_per_block - 1) / threads_per_block;

  // Creating a stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_streams_cuda", ([&] {
    min_reduce_streams_kernel<scalar_t><<<num_blocks, threads_per_block, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        outer,
        r,
        inner);
  }));

  // Synchronizing the stream to ensure completion
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction using CUDA streams (CUDA)");
}