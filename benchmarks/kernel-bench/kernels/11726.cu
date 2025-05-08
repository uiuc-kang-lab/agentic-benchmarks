#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fast exponential approximation (12-bit precision)
__device__ __forceinline__ float fast_exp(float x) {
    x = 1.0f + x / 4096.0f;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    return x;
}

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    // Grid-stride loop setup
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float partial_sums[];
    
    // Vector processing (4 elements per instruction)
    const int n4 = n / 4;
    const float4* logp_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets);
    
    float sum = 0.0f;
    
    // Process vectors with unrolled memory accesses
    int vec_idx = idx;
    while (vec_idx < n4) {
        const float4 logp = __ldg(&logp_vec[vec_idx]);
        const float4 targ = __ldg(&targ_vec[vec_idx]);
        sum += fast_exp(logp.x) - targ.x * logp.x
             + fast_exp(logp.y) - targ.y * logp.y
             + fast_exp(logp.z) - targ.z * logp.z
             + fast_exp(logp.w) - targ.w * logp.w;
        vec_idx += gridDim.x * blockDim.x;
    }

    // Process remaining elements
    int scalar_idx = vec_idx * 4;
    while (scalar_idx < n) {
        sum += fast_exp(log_predictions[scalar_idx]) - targets[scalar_idx] * log_predictions[scalar_idx];
        scalar_idx += gridDim.x * blockDim.x;
    }

    // Shared memory reduction with full unrolling
    partial_sums[threadIdx.x] = sum;
    __syncthreads();

    #pragma unroll
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads*4 - 1) / (threads*4);
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
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