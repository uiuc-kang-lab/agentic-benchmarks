#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline float compute_kldiv_value(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ inline float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    // Shared memory for block-level reduction
    __shared__ float shared[32]; // One element per warp
    float thread_sum = 0.0f;
    
    // Vector processing for aligned data
    const int vec_count = ((gid < n) ? (n - gid) : 0) / 4;
    int idx = gid;
    
    // Process groups of 4 elements using vectorized loads
    if (vec_count > 0) {
        for (int i = 0; i < vec_count; i++, idx += stride * 4) {
            if (idx + 3 >= n) break;
            float4 log_vec = reinterpret_cast<const float4*>(log_predictions + idx)[0];
            float4 target_vec = reinterpret_cast<const float4*>(targets + idx)[0];
            
            thread_sum += compute_kldiv_value(log_vec.x, target_vec.x)
                       + compute_kldiv_value(log_vec.y, target_vec.y)
                       + compute_kldiv_value(log_vec.z, target_vec.z)
                       + compute_kldiv_value(log_vec.w, target_vec.w);
        }
    }
    
    // Process remaining elements
    for (; idx < n; idx += stride) {
        thread_sum += compute_kldiv_value(log_predictions[idx], targets[idx]);
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce(thread_sum);
    
    // Write warp results to shared memory
    const int warp_id = tid / 32;
    if ((tid & 31) == 0) {
        shared[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction using first warp
    if (tid < 32) {
        float warp_sum = (tid < (blockDim.x / 32)) ? shared[tid] : 0.0f;
        warp_sum = warp_reduce(warp_sum);
        
        if (tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward_optimized(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    
    optimized_kl_div_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_optimized, "Optimized KL divergence forward (CUDA)");
}