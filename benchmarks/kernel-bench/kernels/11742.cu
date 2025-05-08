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

    // Process scarlar remainder with unrolled stride
    int scalar_idx = n4 * 4 + global_idx;
    while (scalar_idx < n) {
        float log_pred = __ldg(&log_predictions[scalar_idx]);
        float target = __ldg(&targets[scalar_idx]);
        sum += expf(log_pred) - target * log_pred;
        scalar_idx += gridDim.x * blockDim.x;
    }

    // Manually unrolled warp reduction
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    // Store warp sums in shared memory
    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    // Unrolled block reduction
    if (warp_id == 0) {
        float val = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        
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
    const int blocks = (n + threads*4 - 1) / (threads*4);
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
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA Unrolled Reduction)");
}