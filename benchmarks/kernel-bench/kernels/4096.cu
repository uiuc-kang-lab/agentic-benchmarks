#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses warp-level primitives (__shfl_down_sync and __shfl_sync) 
// to perform a small reduction in each warp. Each thread sets a flag indicating 
// if its value is negative. The warp then reduces these flags; if all threads in 
// the warp have positive values, the expensive expf computation is bypassed for all 
// threads in that warp.
__global__ void elu_warp_opt_kernel(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const unsigned int active_mask = 0xffffffff;

    for (int i = idx; i < n; i += stride) {
        float val = x[i];
        // Set flag: 0 if value is positive, 1 if negative
        int flag = (val > 0.0f) ? 0 : 1;
        
        // Warp-level reduction using __shfl_down_sync to sum flags within a warp
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            flag += __shfl_down_sync(active_mask, flag, offset);
        }

        // Broadcast the reduced flag from lane 0 to all threads in the warp
        int warp_negative_count = __shfl_sync(active_mask, flag, 0);

        // If no thread in the warp had a negative value, use fast path
        if (warp_negative_count == 0) {
            out[i] = val;
        } else {
            // Otherwise, compute ELU individually
            out[i] = (val > 0.0f) ? val : alpha * (expf(val) - 1.0f);
        }
    }
}

// CUDA wrapper that launches the warp-optimized ELU kernel
torch::Tensor elu_warp_opt_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;

    elu_warp_opt_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_warp_opt_cuda, "ELU activation with warp-level optimization (CUDA)");
}
