#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Warp-level reduction using shuffle intrinsics
template <typename scalar_t>
__device__ inline scalar_t warpReduceMin(scalar_t val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    scalar_t tmp = __shfl_down_sync(0xffffffff, val, offset);
    val = (tmp < val) ? tmp : val;
  }
  return val;
}

template <typename scalar_t>
__global__ void warp_min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {
  
  const int idx = blockIdx.x;
  if (idx >= outer * inner) return;

  const int outer_idx = idx / inner;
  const int inner_idx = idx % inner;
  const scalar_t* in_ptr = input + outer_idx * (r * inner) + inner_idx;

  scalar_t local_min = std::numeric_limits<scalar_t>::max();

  // Process r elements with stride equal to blockDim.x
  for (int j = threadIdx.x; j < r; j += blockDim.x) {
    scalar_t val = in_ptr[j * inner];
    if (val < local_min) local_min = val;
  }

  // Warp-level reduction
  local_min = warpReduceMin(local_min);

  // First thread writes final result
  if (threadIdx.x == 0) {
    output[idx] = local_min;
  }
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  input = input.contiguous();

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  int outer = 1;
  for (int i = 0; i < dim; i++) outer *= input.size(i);
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) inner *= input.size(i);

  auto output = torch::empty({outer * inner}, input.options());

  int total = outer * inner;
  const int threads = 32;  // Single warp per block
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "warp_reduce_min", ([&] {
    warp_min_reduction_kernel<scalar_t><<<blocks, threads, 0, 
      c10::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        outer,
        r,
        inner);
  }));

  return output.view(input.sizes().vec().erase(input.sizes().begin() + dim));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction using warp shuffle operations");
}
