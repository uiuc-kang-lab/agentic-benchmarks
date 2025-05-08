#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void warp_min_reduce_shuffle(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  const int idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (idx >= outer || inner_idx >= inner) return;

  const scalar_t* in_ptr = input + idx * (r * inner) + inner_idx;
  
  scalar_t min_val = std::numeric_limits<scalar_t>::max();
  for (int j = threadIdx.x; j < r; j += blockDim.x) {
    min_val = min(min_val, in_ptr[j * inner]);
  }

  __shared__ scalar_t smin[256];
  smin[threadIdx.x] = min_val;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset >= 32; offset >>= 1) {
    if (threadIdx.x < offset) {
      smin[threadIdx.x] = min(smin[threadIdx.x], smin[threadIdx.x + offset]);
    }
    __syncthreads();
  }

  scalar_t val = smin[threadIdx.x];
  if (threadIdx.x < 32) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
  }

  if (threadIdx.x == 0) {
    output[idx * inner + inner_idx] = val;
  }
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
  input = input.contiguous();

  const int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dimension");

  int outer = 1;
  for (int i = 0; i < dim; ++i) outer *= input.size(i);
  const int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; ++i) inner *= input.size(i);

  auto output = torch::empty({outer, inner}, input.options());

  const dim3 threads(32, 8);
  const dim3 blocks(
    (outer + threads.y - 1) / threads.y,
    (inner + threads.x - 1) / threads.x
  );

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "warp_reduce_shuffle", ([&] {
    warp_min_reduce_shuffle<scalar_t><<<blocks, threads, 32*sizeof(scalar_t), 
      c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        outer,
        r,
        inner);
  }));

  return output.view(input.sizes().vec().erase(input.sizes().begin() + dim));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized min reduction with warp shuffles");
}
