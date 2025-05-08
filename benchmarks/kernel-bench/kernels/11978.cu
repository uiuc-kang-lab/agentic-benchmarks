#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__inline__ __device__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void triplet_margin_loss_kernel_optimized(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {

    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Shared memory for partial sums
    extern __shared__ float shared_mem[];
    float* sh_pos = shared_mem;
    float* sh_neg = shared_mem + 32; // Only need warp size

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;
    
    // Process elements with vectorized loads
    float4 sum_pos4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum_neg4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    const int offset = batch_idx * feat_size;
    const float4* anchor4 = reinterpret_cast<const float4*>(anchor + offset);
    const float4* positive4 = reinterpret_cast<const float4*>(positive + offset);
    const float4* negative4 = reinterpret_cast<const float4*>(negative + offset);
    
    const int vec_elements = feat_size / 4;
    
    // Stride by number of warps for better memory coalescing
    for (int i = threadIdx.x; i < vec_elements; i += blockDim.x) {
        float4 a4 = __ldg(&anchor4[i]);
        float4 p4 = __ldg(&positive4[i]);
        float4 n4 = __ldg(&negative4[i]);
        
        float4 d_pos, d_neg;
        d_pos.x = a4.x - p4.x;
        d_pos.y = a4.y - p4.y;
        d_pos.z = a4.z - p4.z;
        d_pos.w = a4.w - p4.w;
        
        d_neg.x = a4.x - n4.x;
        d_neg.y = a4.y - n4.y;
        d_neg.z = a4.z - n4.z;
        d_neg.w = a4.w - n4.w;
        
        sum_pos4.x += d_pos.x * d_pos.x;
        sum_pos4.y += d_pos.y * d_pos.y;
        sum_pos4.z += d_pos.z * d_pos.z;
        sum_pos4.w += d_pos.w * d_pos.w;
        
        sum_neg4.x += d_neg.x * d_neg.x;
        sum_neg4.y += d_neg.y * d_neg.y;
        sum_neg4.z += d_neg.z * d_neg.z;
        sum_neg4.w += d_neg.w * d_neg.w;
    }
    
    // Reduce float4 to single float
    float sum_pos = sum_pos4.x + sum_pos4.y + sum_pos4.z + sum_pos4.w;
    float sum_neg = sum_neg4.x + sum_neg4.y + sum_neg4.z + sum_neg4.w;
    
    // Handle remaining elements
    for (int i = vec_elements * 4 + threadIdx.x; i < feat_size; i += blockDim.x) {
        float a = __ldg(&anchor[offset + i]);
        float p = __ldg(&positive[offset + i]);
        float n = __ldg(&negative[offset + i]);
        
        float d_pos = a - p;
        float d_neg = a - n;
        sum_pos += d_pos * d_pos;
        sum_neg += d_neg * d_neg;
    }
    
    // Warp-level reduction
    sum_pos = warp_reduce_sum(sum_pos);
    sum_neg = warp_reduce_sum(sum_neg);
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        sh_pos[warp_id] = sum_pos;
        sh_neg[warp_id] = sum_neg;
    }
    
    __syncthreads();
    
    // First warp reduces results from all warps
    if (warp_id == 0 && lane_id < num_warps) {
        sum_pos = sh_pos[lane_id];
        sum_neg = sh_neg[lane_id];
        
        sum_pos = warp_reduce_sum(sum_pos);
        sum_neg = warp_reduce_sum(sum_neg);
        
        if (lane_id == 0) {
            float loss = sqrtf(sum_pos) - sqrtf(sum_neg) + margin;
            output[batch_idx] = (loss > 0.0f) ? loss : 0.0f;
        }
    }
}

torch::Tensor triplet_margin_loss_cuda_optimized(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");

    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);
    auto output = torch::empty({batch_size}, anchor.options());

    const int threads = 256;  // Multiple of warp size (32)
    const int shared_mem_size = 64 * sizeof(float); // 2 arrays of warp size
    
    triplet_margin_loss_kernel_optimized<<<batch_size, threads, shared_mem_size>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        output.data_ptr<float>(),
        margin,
        batch_size,
        feat_size);
    
    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda_optimized, "Triplet margin loss forward optimized (CUDA)");
}