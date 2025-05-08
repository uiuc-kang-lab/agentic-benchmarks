#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void hardtanh_balanced_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       const int64_t numel,
                                       const float min_val,
                                       const float max_val) {
    // Use 32-thread warps effectively
    const int tid = threadIdx.x;
    const int wid = tid >> 5;  // Warp ID
    const int lane = tid & 31;  // Lane within warp
    
    // Calculate base index for this thread block
    int64_t base_idx = blockIdx.x * blockDim.x * 4;  // Each thread processes 4 elements
    
    // Process 4 elements per thread with coalesced memory access
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int64_t idx = base_idx + tid + i * blockDim.x;
        if (idx < numel) {
            float val = __ldg(&x[idx]);
            out[idx] = fminf(fmaxf(val, min_val), max_val);
        }
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    const int64_t numel = x.numel();
    
    // Use 128 threads per block for better occupancy
    const int threads = 128; // Align to warp size
    // Calculate blocks needed considering 4 elements per thread
    const int blocks = (numel + (threads * 4) - 1) / (threads * 4);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        hardtanh_balanced_kernel<<<blocks, threads>>>(
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
    if (!x.is_cuda()) throw std::invalid_argument("Input must be CUDA tensor");
    return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardTanh balanced workload (CUDA)");
}