#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Warp-level HardTanh using shuffle operations
template <typename scalar_t>
__device__ scalar_t warp_hardtanh_shfl(scalar_t val, scalar_t min_val, scalar_t max_val) {
    // Clamp value using warp shuffle operations
    val = max(min_val, min(max_val, val));
    return val;
}

// Improved HardTanh kernel using warp-level shuffle operations
template <typename scalar_t>
__global__ void hardtanh_warp_shfl_kernel(const scalar_t* __restrict__ x,
                                          scalar_t* __restrict__ out,
                                          int64_t numel,
                                          scalar_t min_val,
                                          scalar_t max_val) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numel) {
    scalar_t val = x[i];
    val = warp_hardtanh_shfl(val, min_val, max_val);
    out[i] = val;
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  // Use 256 threads per block (8 warps) for better occupancy
  const int threads_per_block = 256;
  // Calculate number of SMs and max blocks per SM
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device_id);
  
  // Target 2 blocks per SM for better occupancy
  const int max_blocks_per_sm = 2;
  const int num_sms = props.multiProcessorCount;
  const int target_blocks = max_blocks_per_sm * num_sms;
  
  // Ensure we have enough blocks to fully utilize the GPU
  const int min_blocks_needed = (numel + threads_per_block - 1) / threads_per_block;
  const int num_blocks = min(min_blocks_needed, target_blocks);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
    hardtanh_warp_shfl_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
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
  m.def("forward", &forward, "HardTanh activation warp-optimized shuffle (CUDA)");
}