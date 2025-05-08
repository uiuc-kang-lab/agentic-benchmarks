#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Store the alpha parameter in constant memory, which is frequently accessed by all threads.
__constant__ float dc_alpha;

// Kernel processing bulk elements using vectorized loads/stores (float4) and reading alpha from constant memory
__global__ void elu_kernel_vec_const(const float4* x, float4* out, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 val = x[idx];
        float4 result;
        result.x = (val.x > 0.f) ? val.x : dc_alpha * (expf(val.x) - 1.f);
        result.y = (val.y > 0.f) ? val.y : dc_alpha * (expf(val.y) - 1.f);
        result.z = (val.z > 0.f) ? val.z : dc_alpha * (expf(val.z) - 1.f);
        result.w = (val.w > 0.f) ? val.w : dc_alpha * (expf(val.w) - 1.f);
        out[idx] = result;
    }
}

// Kernel to process any tail elements that do not fit in a group of 4
__global__ void elu_kernel_tail_const(const float* x, float* out, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < n) {
        float val = x[idx];
        out[idx] = (val > 0.f) ? val : dc_alpha * (expf(val) - 1.f);
    }
}

// Host function that copies the alpha value to constant memory and dispatches kernels
torch::Tensor elu_cuda_const(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    int n4 = n / 4;  // Number of float4 groups
    int remainder = n % 4;

    const int threads = 256;
    int blocks_vec = (n4 + threads - 1) / threads;

    // Copy the read-only alpha to constant memory
    cudaMemcpyToSymbol(dc_alpha, &alpha, sizeof(float));

    // Process the vectorized portion using float4 if available
    if (n4 > 0) {
        elu_kernel_vec_const<<<blocks_vec, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            n4
        );
    }

    // Process tail elements with a scalar kernel
    if (remainder > 0) {
        int tail_offset = n4 * 4;
        int blocks_tail = ((remainder) + threads - 1) / threads;
        elu_kernel_tail_const<<<blocks_tail, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            tail_offset,
            n
        );
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_const, "ELU activation using constant memory (CUDA)");
}
