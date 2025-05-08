#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_hybrid(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int global_idx = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float shared_mem[];
    float* warp_sums = shared_mem;
    
    float sum = 0.0f;

    // Vector processing using float4 with improved coalescing
    const int n4 = n / 4;
    const float4* logp_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets);

    #pragma unroll 4
    for (int vec_idx = global_idx; vec_idx < n4; vec_idx += gridDim.x * blockDim.x) {
        float4 logp = __ldg(&logp_vec[vec_idx]);
        float4 targ = __ldg(&targ_vec[vec_idx]);
        
        // Compute exp once and reuse
        float4 exp_logp;
        exp_logp.x = expf(logp.x);
        exp_logp.y = expf(logp.y);
        exp_logp.z = expf(logp.z);
        exp_logp.w = expf(logp.w);
        
        sum += (exp_logp.x - targ.x * logp.x)
             + (exp_logp.y - targ.y * logp.y)
             + (exp_logp.z - targ.z * logp.z)
             + (exp_logp.w - targ.w * logp.w);
    }

    // Process remaining elements
    #pragma unroll 4
    for (int idx = n4 * 4 + global_idx; idx < n; idx += gridDim.x * blockDim.x) {
        float log_pred = __ldg(&log_predictions[idx]);
        float target = __ldg(&targets[idx]);
        float exp_val = expf(log_pred);
        sum += exp_val - target * log_pred;
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Store warp results
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction using first warp
    if (warp_id == 0 && lane < (blockDim.x / 32)) {
        float warp_sum = warp_sums[lane];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (lane == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), 1024);
    const int shared_mem = (threads / 32) * sizeof(float);
    
    kl_div_kernel_hybrid<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA Hybrid Optimized)");
}