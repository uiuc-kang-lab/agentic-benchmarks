#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void warp_shuffle_min_kernel(
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
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;

  const scalar_t* in_ptr = input + outer_idx * (r * inner) + inner_idx;
  scalar_t min_val = std::numeric_limits<scalar_t>::max();

  // Coalesced global memory access with stride
  for (int j = threadIdx.x; j < r; j += blockDim.x) {
    scalar_t val = in_ptr[j * inner];
    if (val < min_val) min_val = val;
  }

  // Warp-level reduction using shuffle
  for (int offset = 16; offset > 0; offset >>= 1) {
    scalar_t tmp = __shfl_down_sync(0xffffffff, min_val, offset);
    min_val = min(min_val, tmp);
  }

  // Cross-warp reduction using shared memory
  __shared__ scalar_t warp_mins[32];
  if (lane_id == 0) {
    warp_mins[warp_id] = min_val;
  }
  __syncthreads();

  // Final reduction in first warp
  if (warp_id == 0) {
    scalar_t val = lane_id < (blockDim.x + 31) / 32 ? warp_mins[lane_id] : std::numeric_limits<scalar_t>::max();
    for (int offset = 16; offset > 0; offset >>= 1) {
      scalar_t tmp = __shfl_down_sync(0xffffffff, val, offset);
      val = min(val, tmp);
    }
    if (threadIdx.x == 0) {
      output[idx] = val;
    }
  }
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) input = input.contiguous();

  const int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  int outer = 1;
  for (int i = 0; i < dim; i++) outer *= input.size(i);
  const int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) inner *= input.size(i);

  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) if (i != dim) output_shape.push_back(input.size(i));
  auto output = torch::empty(output_shape, input.options());

  const int total = outer * inner;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "warp_shuffle_min", ([&] {
    warp_shuffle_min_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        outer,
        r,
        inner);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Warp shuffle min reduction (CUDA)");
}