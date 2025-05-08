#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float compute_kldiv(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kl_div_kernel_register_optimized(
    const float* __restrict__ log_preds,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gsize = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    
    // Process 4 elements per iteration with manual unrolling
    const int n4 = n / 4;
    for (int i = bid * blockDim.x + tid; i < n4; i += gsize) {
        const float4 log_vec = *reinterpret_cast<const float4*>(log_preds + i*4);
        const float4 targ_vec = *reinterpret_cast<const float4*>(targets + i*4);
        
        sum += compute_kldiv(log_vec.x, targ_vec.x)
             + compute_kldiv(log_vec.y, targ_vec.y)
             + compute_kldiv(log_vec.z, targ_vec.z)
             + compute_kldiv(log_vec.w, targ_vec.w);
    }

    // Handle remaining elements (<= 3)
    const int remainder = n % 4;
    const int tail_start = n4 * 4;
    if (bid == 0 && tid < remainder) {
        sum += compute_kldiv(log_preds[tail_start + tid], targets[tail_start + tid]);
    }

    // Efficient warp reduction with register reuse
    sum = warp_reduce(sum);

    __shared__ float smem[32];
    if ((tid % 32) == 0) {
        smem[tid/32] = sum;
    }
    __syncthreads();

    if (tid < 32) {
        sum = smem[tid];
        sum = warp_reduce(sum);
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward_optimized(
    torch::Tensor log_preds,
    torch::Tensor targets) {
    
    const int n = log_preds.numel();
    auto output = torch::zeros({1}, log_preds.options());

    const int threads = 256;
    const int blocks = min((n + threads*4 - 1) / (threads*4), 1024);

    kl_div_kernel_register_optimized<<<blocks, threads>>>(
        log_preds.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_optimized, "Register optimized KLDiv forward (CUDA)");
}