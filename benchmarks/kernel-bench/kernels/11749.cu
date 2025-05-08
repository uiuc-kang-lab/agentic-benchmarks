#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with manual unrolling via #pragma unroll for critical loops
__global__ void kl_div_kernel_unroll(
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

    const int n4 = n / 4;  // Number of float4 elements
    const float4* logp_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets);

    // Vectorized processing using grid-stride loop with unrolling directive
    for (int vec_idx = global_idx; vec_idx < n4; vec_idx += gridDim.x * blockDim.x) {
        #pragma unroll
        {
            float4 logp = __ldg(&logp_vec[vec_idx]);
            float4 targ = __ldg(&targ_vec[vec_idx]);
            sum += expf(logp.x) - targ.x * logp.x;
            sum += expf(logp.y) - targ.y * logp.y;
            sum += expf(logp.z) - targ.z * logp.z;
            sum += expf(logp.w) - targ.w * logp.w;
        }
    }

    // Process remaining scalar elements
    for (int scalar_idx = n4 * 4 + global_idx; scalar_idx < n; scalar_idx += gridDim.x * blockDim.x) {
        #pragma unroll
        {
            float lp = __ldg(&log_predictions[scalar_idx]);
            float tt = __ldg(&targets[scalar_idx]);
            sum += expf(lp) - tt * lp;
        }
    }

    // Warp-level reduction using unroll pragma
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces block's partial sums
    if (warp_id == 0) {
        float block_sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// CUDA forward function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);
    const int shared_mem = warps_per_block * sizeof(float);

    kl_div_kernel_unroll<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA, unroll pragma optimization)");
}
