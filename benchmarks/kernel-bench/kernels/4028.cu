#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses grid-stride loops with vectorized loads/stores using float4.
// No atomic operations are used because each thread computes a unique output element, eliminating race conditions.
__global__ void elu_kernel_gridstride(const float4* input, float4* output, float alpha, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 val = input[i];
        float4 res;
        res.x = (val.x > 0.f) ? val.x : alpha * (expf(val.x) - 1.f);
        res.y = (val.y > 0.f) ? val.y : alpha * (expf(val.y) - 1.f);
        res.z = (val.z > 0.f) ? val.z : alpha * (expf(val.z) - 1.f);
        res.w = (val.w > 0.f) ? val.w : alpha * (expf(val.w) - 1.f);
        output[i] = res;
    }
}

// Kernel to process the remaining elements that are not a multiple of 4
__global__ void elu_kernel_remainder(const float* input, float* output, float alpha, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        float val = input[i];
        output[i] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
    }
}

// Host interface function
// It splits the work between vectorized and scalar processing using grid-stride loops.
// Atomic operations are not used because each thread writes to a separate location, avoiding contention.
torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    int n4 = n / 4;      // number of float4 elements
    int remainder = n % 4;  // remaining elements

    const int threads = 256;
    int blocks = (n4 + threads - 1) / threads;

    if (n4 > 0) {
        elu_kernel_gridstride<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            alpha,
            n4
        );
    }

    // Process remaining tail elements if any
    if (remainder > 0) {
        int offset = n4 * 4;
        int blocks_remainder = (remainder + threads - 1) / threads;
        elu_kernel_remainder<<<blocks_remainder, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            alpha,
            offset,
            n
        );
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda, "ELU activation with grid-stride loops and minimal atomic usage (CUDA)");
}
