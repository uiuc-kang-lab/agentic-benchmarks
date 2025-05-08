#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  int idx = blockIdx.x;
  if (idx >= outer * inner) return;

  int outer_idx = idx / inner;
  int inner_idx = idx % inner;
  const scalar_t* in_ptr = input + outer_idx * (r * inner) + inner_idx;

  scalar_t local_min = std::numeric_limits<scalar_t>::max();
  for (int j = threadIdx.x; j < r; j += blockDim.x) {
    scalar_t val = in_ptr[j * inner];
    if (val < local_min)
      local_min = val;
  }

  extern __shared__ char shared_mem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);
  sdata[threadIdx.x] = local_min;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s) {
      scalar_t other = sdata[threadIdx.x + s];
      sdata[threadIdx.x] = min(sdata[threadIdx.x], other);
    }
    __syncthreads();
  }
  
  if (threadIdx.x < 32) {
    scalar_t v = sdata[threadIdx.x];
    v = min(v, sdata[threadIdx.x + 32]);
    v = min(v, sdata[threadIdx.x + 16]);
    v = min(v, sdata[threadIdx.x + 8]);
    v = min(v, sdata[threadIdx.x + 4]);
    v = min(v, sdata[threadIdx.x + 2]);
    v = min(v, sdata[threadIdx.x + 1]);
    sdata[threadIdx.x] = v;
  }

  if (threadIdx.x == 0) {
    output[outer_idx * inner + inner_idx] = sdata[0];
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

  auto output = torch::empty({outer, inner}, input.options());

  int total = outer * inner;
  int threads = 256;
  
  // Dynamic block size selection based on reduction dimension
  if (r > 4096) threads = 512;
  else if (r > 2048) threads = 256;
  else if (r > 1024) threads = 128;
  else if (r > 256) threads = 64;
  else if (r > 64) threads = 32;

  threads = max(32, min(1024, threads));

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_cuda", ([&] {
    int shared_mem = threads * sizeof(scalar_t);
    min_reduction_kernel<scalar_t><<<total, threads, shared_mem, c10::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        outer,
        r,
        inner);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Dynamic block size min reduction (CUDA)");
}
