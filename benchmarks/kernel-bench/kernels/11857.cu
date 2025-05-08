#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

__global__ void kldiv_warp_reduce_kernel(
    const float* __restrict__ log_pred,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t n) {

    const int tid = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    const unsigned mask = 0xffffffff;
    float sum = 0.0f;

    // Process 4 elements per thread with vectorized loads
    for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x * 4) {
        float4 pred = *reinterpret_cast<const float4*>(&log_pred[i]);
        float4 t = *reinterpret_cast<const float4*>(&targets[i]);
        
        sum += expf(pred.x) - t.x * pred.x;
        sum += expf(pred.y) - t.y * pred.y;
        sum += expf(pred.z) - t.z * pred.z;
        sum += expf(pred.w) - t.w * pred.w;
    }

    // Complete warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(mask, sum, offset);

    // First lane in warp writes partial sum
    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {
    
    const int64_t n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // H100 optimized config: 256 threads, 576 blocks (144 SMs * 4)
    const int threads = 256;
    const int blocks = 576;

    // Handle any tail elements by processing full float4 groups
    const int64_t aligned_n = (n + 3) & ~3;
    
    kldiv_warp_reduce_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        aligned_n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence warp-reduced forward");
}