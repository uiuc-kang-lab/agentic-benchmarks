#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Kernel optimized to minimize warp divergence
__global__ void hardtanh_warp_divergence_minimized_kernel(const float* __restrict__ x,
                                                           float* __restrict__ out,
                                                           int64_t numel,
                                                           float min_val,
                                                           float max_val) {
    extern __shared__ float shared_data[];
    
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * blockDim.x;
    const int grid_stride = blockDim.x * gridDim.x;
    
    // Process multiple chunks per block to maximize utilization
    for (int base_idx = block_offset; base_idx < numel; base_idx += grid_stride) {
        int idx = base_idx + tid;
        
        // Load chunk into shared memory
        if (idx < numel) {
            shared_data[tid] = __ldg(&x[idx]);
        }
        __syncthreads();
        
        // Process data in shared memory
        if (idx < numel) {
            float val = shared_data[tid];
            float clamped_val = fminf(fmaxf(val, min_val), max_val);
            out[idx] = clamped_val;
        }
        __syncthreads();
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
    hardtanh_warp_divergence_minimized_kernel<<<blocks, threads>>>(
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
  if (!x.is_cuda()) throw std::invalid_argument("Input tensor must be CUDA");
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh CUDA optimized");
}
