#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline float compute_kldiv_value(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ inline float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_optimized_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    int vec_count = n / 4;
    int tail_start = vec_count * 4;
    float sum = 0.0f;

    // Vector processing (aligned)
    for (int i = gid; i < vec_count; i += stride) {
        float4 log_vec = reinterpret_cast<const float4*>(log_predictions)[i];
        float4 target_vec = reinterpret_cast<const float4*>(targets)[i];
        sum += compute_kldiv_value(log_vec.x, target_vec.x)
             + compute_kldiv_value(log_vec.y, target_vec.y)
             + compute_kldiv_value(log_vec.z, target_vec.z)
             + compute_kldiv_value(log_vec.w, target_vec.w);
    }

    // Scalar tail processing (unaligned)
    for (int idx = gid + tail_start; idx < n; idx += stride)
        sum += compute_kldiv_value(log_predictions[idx], targets[idx]);

    // Warp reduction
    sum = warp_reduce(sum);

    __shared__ float shared[32];
    int warp_id = tid / 32;
    if ((tid & 31) == 0)
        shared[warp_id] = sum;
    __syncthreads();

    // Final block reduction
    if (tid < blockDim.x / 32) {
        sum = shared[tid];
        sum = warp_reduce(sum);
        if (tid == 0)
            atomicAdd(output, sum);
    }
}

torch::Tensor kl_div_cuda_optimized(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = std::min((n + threads * 4 - 1) / (threads * 4), 1024);

    kl_div_optimized_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_optimized, "Optimized KLDiv forward (CUDA)");
}