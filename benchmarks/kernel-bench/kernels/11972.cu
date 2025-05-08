#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel optimized using warp-level primitive __shfl_down_sync for reduction
__global__ void triplet_margin_loss_kernel_warp_shfl(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Each block processes one batch element
    int offset = batch_idx * feat_size;
    int tid = threadIdx.x;
    float sum_pos = 0.f;
    float sum_neg = 0.f;

    // Vectorized load using float4 for aligned memory access
    int vectorized_end = (feat_size / 4) * 4;
    const float4* anchor_vec = reinterpret_cast<const float4*>(anchor + offset);
    const float4* positive_vec = reinterpret_cast<const float4*>(positive + offset);
    const float4* negative_vec = reinterpret_cast<const float4*>(negative + offset);
    int num_vec = vectorized_end / 4;

    for (int i = tid; i < num_vec; i += blockDim.x) {
        float4 a = __ldg(&anchor_vec[i]);
        float4 p = __ldg(&positive_vec[i]);
        float4 n = __ldg(&negative_vec[i]);
        float d;
        // Accumulate squared differences for positive
        d = a.x - p.x; sum_pos += d * d;
        d = a.y - p.y; sum_pos += d * d;
        d = a.z - p.z; sum_pos += d * d;
        d = a.w - p.w; sum_pos += d * d;
        
        // Accumulate squared differences for negative
        d = a.x - n.x; sum_neg += d * d;
        d = a.y - n.y; sum_neg += d * d;
        d = a.z - n.z; sum_neg += d * d;
        d = a.w - n.w; sum_neg += d * d;
    }

    // Process remaining elements
    for (int i = vectorized_end + tid; i < feat_size; i += blockDim.x) {
        float a = __ldg(anchor + offset + i);
        float p = __ldg(positive + offset + i);
        float n = __ldg(negative + offset + i);
        float d = a - p;
        sum_pos += d * d;
        d = a - n;
        sum_neg += d * d;
    }

    // Intra-warp reduction using __shfl_down_sync
    unsigned int warp_mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum_pos += __shfl_down_sync(warp_mask, sum_pos, offset);
        sum_neg += __shfl_down_sync(warp_mask, sum_neg, offset);
    }

    // Each warp's lane 0 holds the partial sum
    __shared__ float shared_pos[32];
    __shared__ float shared_neg[32];
    int lane = tid % warpSize;
    int warpId = tid / warpSize;
    if (lane == 0) {
        shared_pos[warpId] = sum_pos;
        shared_neg[warpId] = sum_neg;
    }
    __syncthreads();

    // Final reduction: only the first numWarps threads participate
    int numWarps = blockDim.x / warpSize; // assuming blockDim.x is a multiple of warpSize
    if (tid < numWarps) {
        float final_sum_pos = shared_pos[tid];
        float final_sum_neg = shared_neg[tid];
        // Use warp-level reduction over the participating warp leaders
        for (int off = numWarps / 2; off > 0; off /= 2) {
            final_sum_pos += __shfl_down_sync(warp_mask, final_sum_pos, off);
            final_sum_neg += __shfl_down_sync(warp_mask, final_sum_neg, off);
        }
        if (tid == 0) {
            float loss = sqrtf(final_sum_pos) - sqrtf(final_sum_neg) + margin;
            output[batch_idx] = (loss > 0.f) ? loss : 0.f;
        }
    }
}

// CUDA launcher function
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

    int threads = 256; // Use 256 threads per block
    // Launch one block per batch element
    triplet_margin_loss_kernel_warp_shfl<<<batch_size, threads>>>(
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
    m.def("forward", &triplet_margin_loss_cuda_optimized, "Triplet margin loss forward optimized with warp shfl reduction (CUDA)");
}
