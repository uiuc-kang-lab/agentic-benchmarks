#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel computes the ELU activation in an element-wise manner. In addition,
// it uses warp-level reduction with __shfl_down_sync to determine if all active threads
// in the warp have positive values. If they do, the kernel avoids the extra exponential
// computation by directly assigning the input to output. Otherwise, the standard ELU formula
// is applied. This replaces shared memory based reductions with warp-level primitives,
// reducing runtime overhead on high-performance NVIDIA H100 GPUs.

__global__ void elu_warp_kernel(const float* __restrict__ x, 
                                float* __restrict__ out, 
                                float alpha, 
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Loop over all elements assigned to the thread
    for (int i = idx; i < n; i += stride) {
        float xi = x[i];
        // Flag is 1 if xi > 0, else 0
        float flag = (xi > 0.f) ? 1.f : 0.f;

        // Use warp-level reduction via __shfl_down_sync to sum the flags in the warp
        unsigned int active = __activemask();
        float sum = flag;
        // Perform reduction: assume warpSize is 32
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(active, sum, offset);
        }
        // Broadcast the reduction result from lane 0 to all lanes in the warp
        float warp_sum = __shfl_sync(active, sum, 0);
        // Count of active lanes in the warp
        int active_count = __popc(active);

        // If all active threads in the warp have positive values, simply pass xi
        // Otherwise, compute ELU: xi for positive values, alpha*(exp(xi)-1) for negatives
        if (warp_sum == float(active_count)) {
            out[i] = xi;
        } else {
            out[i] = (xi > 0.f) ? xi : alpha * (expf(xi) - 1.f);
        }
    }
}

// CUDA wrapper function
torch::Tensor elu_warp_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    elu_warp_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_warp_cuda, "Warp-level ELU activation using warp-level reduction (CUDA)");
}
