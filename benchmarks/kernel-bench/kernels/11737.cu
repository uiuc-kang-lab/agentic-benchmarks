#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int global_idx = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float warp_sums[];
    
    float sum = 0.0f;

    // Vector processing using float4
    const int n4 = n / 4;
    const float4* logp_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets);

    int vec_idx = global_idx;
    while (vec_idx < n4) {
        float4 logp = __ldg(&logp_vec[vec_idx]);
        float4 targ = __ldg(&targ_vec[vec_idx]);
        sum += expf(logp.x) - targ.x * logp.x
             + expf(logp.y) - targ.y * logp.y
             + expf(logp.z) - targ.z * logp.z
             + expf(logp.w) - targ.w * logp.w;
        vec_idx += gridDim.x * blockDim.x;
    }

    // Scalar processing for remainder
    int scalar_idx = n4 * 4 + global_idx;
    while (scalar_idx < n) {
        float log_pred = __ldg(&log_predictions[scalar_idx]);
        float target = __ldg(&targets[scalar_idx]);
        sum += expf(log_pred) - target * log_pred;
        scalar_idx += gridDim.x * blockDim.x;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    // Store warp sums in shared memory
    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    // First warp reduces final block sum
    if (warp_id == 0) {
        float val = lane < (blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        
        if (lane == 0)
            atomicAdd(output, val);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks = min((n + threads*4 - 1) / (threads*4), 1024);
    const int shared_mem = warps_per_block * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA Warp Optimized)");
}