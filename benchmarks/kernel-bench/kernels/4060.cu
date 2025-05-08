#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

/* Optimized ELU kernel with corrected vectorized load and index handling. Each thread processes ELEMENTS_PER_THREAD consecutive elements. */
__global__ void optimized_elu_kernel(const float* __restrict__ x, 
                                      float* __restrict__ out,
                                      float alpha, 
                                      int n) {
    int tid = threadIdx.x;
    // Each thread processes ELEMENTS_PER_THREAD consecutive elements
    int base = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + tid * ELEMENTS_PER_THREAD;

    if (base + ELEMENTS_PER_THREAD <= n) {
        // Use vectorized load when there are enough elements
        float4 in_val = reinterpret_cast<const float4*>(x)[base / 4];
        float4 res;
        res.x = (in_val.x > 0.f) ? in_val.x : alpha * (expf(in_val.x) - 1.f);
        res.y = (in_val.y > 0.f) ? in_val.y : alpha * (expf(in_val.y) - 1.f);
        res.z = (in_val.z > 0.f) ? in_val.z : alpha * (expf(in_val.z) - 1.f);
        res.w = (in_val.w > 0.f) ? in_val.w : alpha * (expf(in_val.w) - 1.f);
        reinterpret_cast<float4*>(out)[base / 4] = res;
    } else {
        // Fallback for tail elements
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int idx = base + i;
            if (idx < n) {
                float val = x[idx];
                out[idx] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
            }
        }
    }
}

torch::Tensor optimized_elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = BLOCK_SIZE;
    const int blocks = (n + (threads * ELEMENTS_PER_THREAD) - 1) / (threads * ELEMENTS_PER_THREAD);
    
    optimized_elu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        alpha,
        n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_elu_cuda, "Optimized ELU activation (CUDA)");
}