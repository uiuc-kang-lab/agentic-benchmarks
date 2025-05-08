#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename scalar_t>
__global__ void hardtanh_kernel_shared(const scalar_t* __restrict__ x,
                                        scalar_t* __restrict__ out,
                                        int64_t numel,
                                        scalar_t min_val,
                                        scalar_t max_val) {
  extern __shared__ scalar_t shared_mem[];
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  for (int64_t idx = i; idx < numel; idx += stride) {
    scalar_t val = x[idx];
    shared_mem[threadIdx.x] = val;
    __syncthreads();

    // Clamp between min_val and max_val using shared memory.
    if (shared_mem[threadIdx.x] < min_val) {
      shared_mem[threadIdx.x] = min_val;
    } else if (shared_mem[threadIdx.x] > max_val) {
      shared_mem[threadIdx.x] = max_val;
    }
    
    __syncthreads();
    out[idx] = shared_mem[threadIdx.x];
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
    hardtanh_kernel_shared<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
        x.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        numel,
        static_cast<scalar_t>(min_val),
        static_cast<scalar_t>(max_val)
    );
  }));

  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) {
    throw std::invalid_argument("Input tensor must be a CUDA tensor");
  }
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh activation (CUDA)");
}