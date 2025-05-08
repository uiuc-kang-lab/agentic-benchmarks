#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_atomic_minimization(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;

    // Vector processing using float4 for 128-bit aligned accesses
    const int n4 = n / 4;
    const float4* logp_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets);

    // Process vector elements using __ldg for read-only cache
    int vec_idx = idx;
    while (vec_idx < n4) {
        float4 logp = __ldg(&logp_vec[vec_idx]);
        float4 targ = __ldg(&targ_vec[vec_idx]);
        sum += expf(logp.x) - targ.x * logp.x
             + expf(logp.y) - targ.y * logp.y
             + expf(logp.z) - targ.z * logp.z
             + expf(logp.w) - targ.w * logp.w;
        vec_idx += gridDim.x * blockDim.x;
    }

    // Process remaining elements using scalar __ldg
    int scalar_idx = n4 * 4 + idx;
    while (scalar_idx < n) {
        float log_pred = __ldg(&log_predictions[scalar_idx]);
        float target = __ldg(&targets[scalar_idx]);
        sum += expf(log_pred) - target * log_pred;
        scalar_idx += gridDim.x * blockDim.x;
    }

    // Store partial sum in shared memory
    partial_sums[threadIdx.x] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Minimize atomic operations by only using them once per block
    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

// Optimized combined version of kl_div_cuda_forward

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n / 4 + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_atomic_minimization<<<blocks, threads, shared_mem>>>(
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
