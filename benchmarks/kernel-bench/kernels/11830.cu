#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float compute_kldiv(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float process_float4(float4 log_vec, float4 target_vec) {
    return compute_kldiv(log_vec.x, target_vec.x) +
           compute_kldiv(log_vec.y, target_vec.y) +
           compute_kldiv(log_vec.z, target_vec.z) +
           compute_kldiv(log_vec.w, target_vec.w);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_unrolled(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    // Process 16 elements per iteration (4 float4 vectors)
    float local_sum = 0.0f;
    const int vec_count = n / 16;  // Number of complete 16-element groups
    
    #pragma unroll 1
    for (int i = gid; i < vec_count; i += stride) {
        const int base_idx = i * 16;
        const float4* log_ptr = reinterpret_cast<const float4*>(log_predictions + base_idx);
        const float4* target_ptr = reinterpret_cast<const float4*>(targets + base_idx);

        // Manually unroll 4x float4 processing
        float4 log_vec1 = log_ptr[0];
        float4 target_vec1 = target_ptr[0];
        float4 log_vec2 = log_ptr[1];
        float4 target_vec2 = target_ptr[1];
        float4 log_vec3 = log_ptr[2];
        float4 target_vec3 = target_ptr[2];
        float4 log_vec4 = log_ptr[3];
        float4 target_vec4 = target_ptr[3];

        local_sum += process_float4(log_vec1, target_vec1);
        local_sum += process_float4(log_vec2, target_vec2);
        local_sum += process_float4(log_vec3, target_vec3);
        local_sum += process_float4(log_vec4, target_vec4);
    }

    // Efficient block reduction with unrolled warp reduction
    __shared__ float warp_sums[32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Warp-level reduction
    local_sum = warp_reduce_sum(local_sum);

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        local_sum = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        
        if (lane_id == 0) {
            atomicAdd(output, local_sum);
        }
    }

    // Handle remaining elements (not part of complete 16-element groups)
    if (blockIdx.x == 0 && tid == 0) {
        const int remaining_start = vec_count * 16;
        float tail_sum = 0.0f;
        
        #pragma unroll 4
        for (int i = remaining_start; i < n; i++) {
            tail_sum += compute_kldiv(log_predictions[i], targets[i]);
        }
        
        atomicAdd(output, tail_sum);
    }
}

torch::Tensor kl_div_cuda_forward_unrolled(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int vec_groups = n / 16;  // Number of 16-element groups
    const int blocks = min((vec_groups + threads - 1) / threads, 1024);

    kl_div_kernel_unrolled<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_unrolled, "Unrolled KLDiv forward (CUDA)");
}