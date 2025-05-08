#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void optimized_smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n_elements
) {
    // Use vectorized loads for better memory bandwidth
    using float4_t = float4;
    
    // Shared memory for reduction
    __shared__ float shared_sum[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    float thread_sum = 0.0f;
    
    // Vector load processing - 4 elements at a time
    const float4_t* pred4 = reinterpret_cast<const float4_t*>(predictions);
    const float4_t* targ4 = reinterpret_cast<const float4_t*>(targets);
    const int vec_elements = n_elements / 4;
    
    #pragma unroll 2
    for (int i = gid; i < vec_elements; i += stride) {
        float4_t pred_vec = __ldg(&pred4[i]);
        float4_t targ_vec = __ldg(&targ4[i]);
        
        float diff0 = pred_vec.x - targ_vec.x;
        float diff1 = pred_vec.y - targ_vec.y;
        float diff2 = pred_vec.z - targ_vec.z;
        float diff3 = pred_vec.w - targ_vec.w;
        
        float abs_diff0 = fabsf(diff0);
        float abs_diff1 = fabsf(diff1);
        float abs_diff2 = fabsf(diff2);
        float abs_diff3 = fabsf(diff3);
        
        thread_sum += (abs_diff0 < 1.0f) ? 0.5f * diff0 * diff0 : abs_diff0 - 0.5f;
        thread_sum += (abs_diff1 < 1.0f) ? 0.5f * diff1 * diff1 : abs_diff1 - 0.5f;
        thread_sum += (abs_diff2 < 1.0f) ? 0.5f * diff2 * diff2 : abs_diff2 - 0.5f;
        thread_sum += (abs_diff3 < 1.0f) ? 0.5f * diff3 * diff3 : abs_diff3 - 0.5f;
    }
    
    // Handle remaining elements
    const int rem_start = vec_elements * 4;
    for (int i = rem_start + gid; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }
    
    // Store in shared memory
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Two-level reduction: first warp-level, then block-level
    if (tid < 32) {
        // Warp-level reduction (unrolled)
        if (BLOCK_SIZE >= 64) shared_sum[tid] += shared_sum[tid + 32];
        if (BLOCK_SIZE >= 32) shared_sum[tid] += shared_sum[tid + 16];
        if (BLOCK_SIZE >= 16) shared_sum[tid] += shared_sum[tid + 8];
        if (BLOCK_SIZE >= 8) shared_sum[tid] += shared_sum[tid + 4];
        if (BLOCK_SIZE >= 4) shared_sum[tid] += shared_sum[tid + 2];
        if (BLOCK_SIZE >= 2) shared_sum[tid] += shared_sum[tid + 1];
        
        // First thread in warp writes result
        if (tid == 0) {
            atomicAdd(output, shared_sum[0] / n_elements);
        }
    }
}

torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");
    
    const int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());
    
    constexpr int BLOCK_SIZE = 256;
    const int VECTOR_SIZE = 4;
    const int grid_size = std::min(256, (n + BLOCK_SIZE * VECTOR_SIZE - 1) / (BLOCK_SIZE * VECTOR_SIZE));
    
    optimized_smooth_l1_loss_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Optimized Smooth L1 Loss (CUDA)");
}