#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__device__ float4 load_vector4(const float* ptr, int idx) {
    return __ldg(reinterpret_cast<const float4*>(ptr) + idx);
}

__device__ float process_vector_element(const float4& logp, const float4& targ, int component) {
    const float lp = (&logp.x)[component];
    const float tt = (&targ.x)[component];
    return expf(lp) - tt * lp;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float process_scalar_element(const float* logp, const float* targ, int idx) {
    float lp = __ldg(logp + idx);
    float tt = __ldg(targ + idx);
    return expf(lp) - tt * lp;
}

} // anonymous namespace

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

    // Vectorized processing
    const int n4 = n / 4;
    int vec_idx = global_idx;
    while (vec_idx < n4) {
        float4 logp = load_vector4(log_predictions, vec_idx);
        float4 targ = load_vector4(targets, vec_idx);
        
        for (int i = 0; i < 4; ++i)
            sum += process_vector_element(logp, targ, i);
        
        vec_idx += gridDim.x * blockDim.x;
    }

    // Scalar processing
    int scalar_idx = n4 * 4 + global_idx;
    while (scalar_idx < n) {
        sum += process_scalar_element(log_predictions, targets, scalar_idx);
        scalar_idx += gridDim.x * blockDim.x;
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // Store warp sums
    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    // Block-level reduction
    if (warp_id == 0) {
        float block_sum = lane < (blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        
        if (lane == 0)
            atomicAdd(output, block_sum);
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
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA Modular Reduction)");
}
