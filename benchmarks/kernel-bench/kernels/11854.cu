#include <torch/extension.h>
#include <cuda_fp16.h>

__global__ void kldiv_sm_optimized_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t n) {

    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int elements_per_block = blockDim.x * 4;
    float sum = 0.0f;

    #pragma unroll 4
    for (int64_t i = tid * 4; i < n; i += gridDim.x * elements_per_block) {
        float4 log_pred = *reinterpret_cast<const float4*>(&log_predictions[i]);
        float4 target = *reinterpret_cast<const float4*>(&targets[i]);
        
        sum += expf(log_pred.x) - target.x * log_pred.x;
        sum += expf(log_pred.y) - target.y * log_pred.y;
        sum += expf(log_pred.z) - target.z * log_pred.z;
        sum += expf(log_pred.w) - target.w * log_pred.w;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Single atomic per block
    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int64_t n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 512;
    const int blocks = 144;  // Directly target H100's 144 SMs

    kldiv_sm_optimized_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "H100-optimized KL divergence");
}