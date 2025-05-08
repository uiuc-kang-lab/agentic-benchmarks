#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

typedef float float4 __attribute__((ext_vector_type(4)));

__device__ inline float4 hinge_loss4(float4 pred, float4 target) {
    float4 result;
    result.x = fmaxf(0.0f, 1.0f - pred.x * target.x);
    result.y = fmaxf(0.0f, 1.0f - pred.y * target.y);
    result.z = fmaxf(0.0f, 1.0f - pred.z * target.z);
    result.w = fmaxf(0.0f, 1.0f - pred.w * target.w);
    return result;
}

__global__ void vectorized_hinge_loss_kernel(const float* __restrict__ predictions,
                                           const float* __restrict__ targets,
                                           float* __restrict__ output,
                                           int n) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;

    // Vectorized processing for bulk elements
    for (unsigned int v_idx = tid; v_idx < n/4; v_idx += stride) {
        float4 pred_vec = *reinterpret_cast<const float4*>(predictions + v_idx*4);
        float4 target_vec = *reinterpret_cast<const float4*>(targets + v_idx*4);
        float4 result_vec = hinge_loss4(pred_vec, target_vec);
        *reinterpret_cast<float4*>(output + v_idx*4) = result_vec;
    }

    // Process remaining elements (scalar fallback)
    const unsigned int remains_start = (n/4)*4;
    for (unsigned int idx = remains_start + tid; idx < n; idx += stride) {
        output[idx] = fmaxf(0.0f, 1.0f - predictions[idx] * targets[idx]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);

    vectorized_hinge_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized Hinge Loss Forward");
}
