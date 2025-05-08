#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void triplet_margin_loss_combined(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int offset = batch_idx * feat_size;
    int tid = threadIdx.x;
    float sum_pos = 0.f;
    float sum_neg = 0.f;

    // Vectorized processing with 128-bit loads
    int vectorized_end = (feat_size / 4) * 4;
    const float4* anchor_vec = reinterpret_cast<const float4*>(anchor + offset);
    const float4* positive_vec = reinterpret_cast<const float4*>(positive + offset);
    const float4* negative_vec = reinterpret_cast<const float4*>(negative + offset);
    int num_vec = vectorized_end / 4;

    for (int i = tid; i < num_vec; i += blockDim.x) {
        float4 a = __ldg(&anchor_vec[i]);
        float4 p = __ldg(&positive_vec[i]);
        float4 n = __ldg(&negative_vec[i]);

        // Positive distances
        float d = a.x - p.x; sum_pos += d * d;
        d = a.y - p.y; sum_pos += d * d;
        d = a.z - p.z; sum_pos += d * d;
        d = a.w - p.w; sum_pos += d * d;

        // Negative distances
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

    // Warp-level reduction
    unsigned int warp_mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_pos += __shfl_down_sync(warp_mask, sum_pos, offset);
        sum_neg += __shfl_down_sync(warp_mask, sum_neg, offset);
    }

    // Cross-warp reduction
    __shared__ float shared_pos[32];
    __shared__ float shared_neg[32];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    
    if (lane == 0) {
        shared_pos[warp_id] = sum_pos;
        shared_neg[warp_id] = sum_neg;
    }
    __syncthreads();

    if (tid < warpSize) {
        sum_pos = tid < blockDim.x / warpSize ? shared_pos[tid] : 0;
        sum_neg = tid < blockDim.x / warpSize ? shared_neg[tid] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_pos += __shfl_down_sync(warp_mask, sum_pos, offset);
            sum_neg += __shfl_down_sync(warp_mask, sum_neg, offset);
        }

        if (tid == 0) {
            float loss = sqrtf(sum_pos) - sqrtf(sum_neg) + margin;
            output[batch_idx] = fmaxf(loss, 0.0f);
        }
    }
}

torch::Tensor triplet_margin_loss_cuda_combined(
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

    int threads = 256;
    triplet_margin_loss_combined<<<batch_size, threads>>>( 
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
    m.def("forward", &triplet_margin_loss_cuda_combined, "Triplet margin loss combined optimized (CUDA)");
}
