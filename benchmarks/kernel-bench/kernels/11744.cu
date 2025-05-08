#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

// Constant memory for frequently used values
__constant__ int WARP_SIZE = 32;
__constant__ unsigned int FULL_MASK = 0xffffffff;

__forceinline__ __device__ float4 load_vector4(const float* ptr, int idx) {
    return __ldg(reinterpret_cast<const float4*>(ptr) + idx);
}

__forceinline__ __device__ float process_element(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;  // Using faster __expf
}

__forceinline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

} // anonymous namespace

template<int BLOCK_SIZE = 256, int VECTOR_SIZE = 4>
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int global_idx = blockIdx.x * BLOCK_SIZE + tid;
    
    extern __shared__ float warp_sums[];
    
    float sum = 0.0f;

    // Vectorized processing with loop unrolling
    const int n4 = n / VECTOR_SIZE;
    int vec_idx = global_idx;
    
    #pragma unroll 2
    while (vec_idx < n4) {
        float4 logp = load_vector4(log_predictions, vec_idx);
        float4 targ = load_vector4(targets, vec_idx);
        
        sum += process_element(logp.x, targ.x);
        sum += process_element(logp.y, targ.y);
        sum += process_element(logp.z, targ.z);
        sum += process_element(logp.w, targ.w);
        
        vec_idx += gridDim.x * BLOCK_SIZE;
    }

    // Scalar processing for remainder using vectorized loads when possible
    int scalar_idx = n4 * VECTOR_SIZE + global_idx;
    while (scalar_idx < n) {
        sum += process_element(
            __ldg(&log_predictions[scalar_idx]),
            __ldg(&targets[scalar_idx])
        );
        scalar_idx += gridDim.x * BLOCK_SIZE;
    }

    // Two-level reduction: warp then block
    sum = warp_reduce_sum(sum);
    
    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane] : 0.0f;
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
    
    constexpr int BLOCK_SIZE = 256;
    const int warps_per_block = BLOCK_SIZE / 32;
    const int blocks = min((n + BLOCK_SIZE*4 - 1) / (BLOCK_SIZE*4), 1024);
    const int shared_mem = warps_per_block * sizeof(float);
    
    kl_div_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA Optimized)");
}