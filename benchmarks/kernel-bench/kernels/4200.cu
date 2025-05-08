#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Kernel using shared memory for reductions
__global__ void hardtanh_kernel_shared_mem(const float* __restrict__ x,
                                            float* __restrict__ out,
                                            int64_t numel,
                                            float min_val,
                                            float max_val) {
  extern __shared__ float shared_mem[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // Load data into shared memory
  if (idx < numel) {
    shared_mem[tid] = __ldg(&x[idx]);
  } else {
    shared_mem[tid] = 0.0f;  // Initialize out of bounds values
  }
  __syncthreads();

  // Perform reduction on shared memory
  if (idx < numel) {
    shared_mem[tid] = fminf(fmaxf(shared_mem[tid], min_val), max_val);
  }
  __syncthreads();

  // Write results back to global memory
  if (idx < numel) {
    out[idx] = shared_mem[tid];
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;
  const size_t shared_mem_size = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
    hardtanh_kernel_shared_mem<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        min_val,
        max_val
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
  m.def("forward", &forward, "HardTanh activation optimized with shared memory (CUDA)");
}