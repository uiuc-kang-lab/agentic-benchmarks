#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32

__global__ void softsign_warp_kernel(const float* __restrict__ x, 
                                   float* __restrict__ out, 
                                   const int num_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int warp_elements = wid * WARP_SIZE;
    
    if (warp_elements < num_elements) {
        const int idx = warp_elements + lane;
        if (idx < num_elements) {
            const float val = __ldg(&x[idx]);
            const float result = val / (1.0f + fabsf(val));
            out[idx] = result;
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / WARP_SIZE;
    const int num_warps = (num_elements + WARP_SIZE - 1) / WARP_SIZE;
    const int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    
    softsign_warp_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized Softsign activation (CUDA)");
}