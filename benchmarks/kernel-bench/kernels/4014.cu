#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel: Utilize __ldg for read-only global memory access and align to 128-bit
__global__ void elu_kernel_ldg(const float4* __restrict__ x, float4* out, float alpha, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 val = __ldg(&x[idx]);
        float4 result;
        result.x = (val.x > 0) ? val.x : alpha * (expf(val.x) - 1);
        result.y = (val.y > 0) ? val.y : alpha * (expf(val.y) - 1);
        result.z = (val.z > 0) ? val.z : alpha * (expf(val.z) - 1);
        result.w = (val.w > 0) ? val.w : alpha * (expf(val.w) - 1);
        out[idx] = result;
    }
}

// Kernel: Process the tail elements that are not a multiple of 4
__global__ void elu_kernel_tail(const float* x, float* out, float alpha, int offset, int n) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (globalIdx < n) {
        float val = x[globalIdx];
        out[globalIdx] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
    }
}

// Host interface function
// It dispatches two kernel calls: one for the vectorized portion and one for any remaining tail elements.
torch::Tensor elu_cuda_ldg(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    // Determine the number of float4 groups
    int n4 = n / 4;            // number of vectorizable groups
    int remainder = n % 4;       // remaining elements

    const int threads = 256;
    int blocks_vec = (n4 + threads - 1) / threads;

    // If there is at least one vectorized element, process it using the shared memory kernel
    if (n4 > 0) {
        elu_kernel_ldg<<<blocks_vec, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            alpha,
            n4
        );
    }

    // Process any remaining tail elements with a scalar kernel
    if (remainder > 0) {
        int tail_offset = n4 * 4;
        int blocks_tail = (remainder + threads - 1) / threads;
        elu_kernel_tail<<<blocks_tail, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            alpha,
            tail_offset,
            n
        );
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_ldg, "ELU activation with __ldg and aligned memory (CUDA)");
}