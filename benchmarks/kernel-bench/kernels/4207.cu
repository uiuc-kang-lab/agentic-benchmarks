#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void hardtanh_warp_kernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    int64_t numel,
                                    float min_val,
                                    float max_val) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid & 0x1F;  // Lane ID within warp
    const unsigned int wid = tid >> 5;        // Warp ID within thread block
    const unsigned int gid = blockIdx.x * blockDim.x + tid;
    const unsigned int warp_size = 32;
    const unsigned int grid_stride = gridDim.x * blockDim.x;

    // Process elements with warp-stride loop
    for (int64_t idx = gid; idx < numel; idx += grid_stride) {
        float val = __ldg(&x[idx]);
        val = fminf(fmaxf(val, min_val), max_val);
        
        // Use warp shuffle to share results within warp
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float shuffled = __shfl_down_sync(0xffffffff, val, offset);
            if (lane_id < offset) {
                val = fminf(fmaxf(val, shuffled), max_val);
            }
        }
        
        // Write result
        out[idx] = val;
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();

    const int threads = 256;  // Multiple of warp size (32)
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        hardtanh_warp_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "HardTanh warp optimized (CUDA)");
}