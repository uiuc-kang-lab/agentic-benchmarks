#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel with manually unrolled grid-stride loop to reduce loop overhead
__global__ void elu_kernel_unroll(const float* x, float* out, float alpha, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per iteration
    for (int i = tid; i < n; i += stride * 4) {
        int idx0 = i;
        int idx1 = i + stride;
        int idx2 = i + 2 * stride;
        int idx3 = i + 3 * stride;
        
        if (idx0 < n) {
            float val = x[idx0];
            out[idx0] = (val > 0.0f) ? val : alpha * (expf(val) - 1.0f);
        }
        if (idx1 < n) {
            float val = x[idx1];
            out[idx1] = (val > 0.0f) ? val : alpha * (expf(val) - 1.0f);
        }
        if (idx2 < n) {
            float val = x[idx2];
            out[idx2] = (val > 0.0f) ? val : alpha * (expf(val) - 1.0f);
        }
        if (idx3 < n) {
            float val = x[idx3];
            out[idx3] = (val > 0.0f) ? val : alpha * (expf(val) - 1.0f);
        }
    }
}

// Host function that launches the unrolled kernel
torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    elu_kernel_unroll<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda, "ELU activation with unrolled loops (CUDA)");
}
