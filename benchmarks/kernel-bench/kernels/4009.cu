#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Branchless vectorized kernel using float4 to avoid warp divergence
__global__ void branchless_elu_kernel_vec4(const float4* x, float4* out, float alpha, int num_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec) {
        float4 in_val = x[idx];
        float4 res;
        // Compute branchless ELU for each component
        res.x = fmaxf(in_val.x, 0.0f) + fminf(alpha * (expf(in_val.x) - 1.0f), 0.0f);
        res.y = fmaxf(in_val.y, 0.0f) + fminf(alpha * (expf(in_val.y) - 1.0f), 0.0f);
        res.z = fmaxf(in_val.z, 0.0f) + fminf(alpha * (expf(in_val.z) - 1.0f), 0.0f);
        res.w = fmaxf(in_val.w, 0.0f) + fminf(alpha * (expf(in_val.w) - 1.0f), 0.0f);
        out[idx] = res;
    }
}

// Tail kernel for remaining elements that don't fit into float4 groups
__global__ void branchless_elu_kernel_tail(const float* x, float* out, float alpha, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < n) {
        float val = x[idx];
        out[idx] = fmaxf(val, 0.0f) + fminf(alpha * (expf(val) - 1.0f), 0.0f);
    }
}

// Host interface function
torch::Tensor branchless_elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    // Process bulk of data in groups of 4 using vectorized operations
    int num_vec = n / 4;  // number of float4 groups
    int remainder = n % 4;
    const int threads = 256;
    if (num_vec > 0) {
        int blocks_vec = (num_vec + threads - 1) / threads;
        branchless_elu_kernel_vec4<<<blocks_vec, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            alpha,
            num_vec
        );
    }
    // Process tail elements that are not multiple of 4
    if (remainder > 0) {
        int tail_offset = num_vec * 4;
        int blocks_tail = (remainder + threads - 1) / threads;
        branchless_elu_kernel_tail<<<blocks_tail, threads>>>(
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
    m.def("forward", &branchless_elu_cuda, "Branchless ELU activation with vectorized load/store (CUDA)");
}
