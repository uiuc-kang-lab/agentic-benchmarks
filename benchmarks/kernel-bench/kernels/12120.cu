#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void hinge_loss_coalesced_kernel(const float* predictions, const float* targets, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_offset = idx * 4;
    if (vec_offset + 3 < n) {
        float4 pred = *reinterpret_cast<const float4*>(predictions + vec_offset);
        float4 targ = *reinterpret_cast<const float4*>(targets + vec_offset);
        float4 result;
        result.x = fmaxf(0.0f, 1.0f - pred.x * targ.x);
        result.y = fmaxf(0.0f, 1.0f - pred.y * targ.y);
        result.z = fmaxf(0.0f, 1.0f - pred.z * targ.z);
        result.w = fmaxf(0.0f, 1.0f - pred.w * targ.w);
        *reinterpret_cast<float4*>(output + vec_offset) = result;
    } else {
        for (int i = 0; i < 4; ++i) {
            if (vec_offset + i < n) {
                output[vec_offset + i] = fmaxf(0.0f, 1.0f - predictions[vec_offset + i] * targets[vec_offset + i]);
            }
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    int threads = 256;
    int elements_per_block = threads * 4;
    int blocks = (n + elements_per_block - 1) / elements_per_block;

    hinge_loss_coalesced_kernel<<<blocks, threads>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), output.data_ptr<float>(), n);
    
    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward (Optimized)");
}
