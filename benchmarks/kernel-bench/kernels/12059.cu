#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

struct __align__(16) Float4 {
    float x, y, z, w;
};

__global__ void hinge_loss_coalesced_kernel(const float4* __restrict__ predictions,
                                           const float4* __restrict__ targets,
                                           float4* __restrict__ output,
                                           int n4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n4) {
        // Load 4 elements at once using vectorized loads
        float4 pred4 = __ldg(&predictions[tid]);
        float4 targ4 = __ldg(&targets[tid]);
        
        // Process four elements
        float4 out4;
        out4.x = fmaxf(0.0f, 1.0f - pred4.x * targ4.x);
        out4.y = fmaxf(0.0f, 1.0f - pred4.y * targ4.y);
        out4.z = fmaxf(0.0f, 1.0f - pred4.z * targ4.z);
        out4.w = fmaxf(0.0f, 1.0f - pred4.w * targ4.w);
        
        // Store 4 elements at once
        output[tid] = out4;
    }
}

__global__ void hinge_loss_remainder_kernel(const float* __restrict__ predictions,
                                          const float* __restrict__ targets,
                                          float* __restrict__ output,
                                          int start_idx,
                                          int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid + start_idx < n) {
        float pred = __ldg(&predictions[tid + start_idx]);
        float targ = __ldg(&targets[tid + start_idx]);
        output[tid + start_idx] = fmaxf(0.0f, 1.0f - pred * targ);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    // Calculate number of float4 elements
    int n4 = n / 4;
    int remainder = n % 4;

    // Process aligned data using float4
    if (n4 > 0) {
        int threads = 256;
        int blocks = (n4 + threads - 1) / threads;

        hinge_loss_coalesced_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(predictions.data_ptr<float>()),
            reinterpret_cast<const float4*>(targets.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            n4
        );
    }

    // Process remaining elements
    if (remainder > 0) {
        int threads = 128;
        int blocks = (remainder + threads - 1) / threads;

        hinge_loss_remainder_kernel<<<blocks, threads>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n4 * 4,
            n
        );
    }

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Memory Access Hinge Loss Forward");
}