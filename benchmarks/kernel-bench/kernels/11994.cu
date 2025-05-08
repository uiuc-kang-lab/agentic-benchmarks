#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Store the margin in constant memory (read-only across all threads).
__constant__ float c_margin;

// Optimized kernel using vectorized loads, warp-level reduction, and constant memory for the margin.
__global__ void triplet_margin_loss_kernel_constopt(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ output,
    const int batch_size,
    const int feat_size) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int offset = batch_idx * feat_size;
    int tid = threadIdx.x;
    float sum_pos = 0.f;
    float sum_neg = 0.f;

    // Use vectorized 128-bit loads with float4 if possible for efficiency
    int vectorized_end = (feat_size / 4) * 4;
    const float4* anchor_vec = reinterpret_cast<const float4*>(anchor + offset);
    const float4* positive_vec = reinterpret_cast<const float4*>(positive + offset);
    const float4* negative_vec = reinterpret_cast<const float4*>(negative + offset);
    int num_vec = vectorized_end / 4;

    // Unrolled vectorized loop for improved throughput
    for (int i = tid; i < num_vec; i += blockDim.x * 2) {
        float4 a = __ldg(&anchor_vec[i]);
        float4 p = __ldg(&positive_vec[i]);
        float4 n = __ldg(&negative_vec[i]);

        float d = a.x - p.x; sum_pos += d * d;
        d = a.y - p.y; sum_pos += d * d;
        d = a.z - p.z; sum_pos += d * d;
        d = a.w - p.w; sum_pos += d * d;

        d = a.x - n.x; sum_neg += d * d;
        d = a.y - n.y; sum_neg += d * d;
        d = a.z - n.z; sum_neg += d * d;
        d = a.w - n.w; sum_neg += d * d;

        int j = i + blockDim.x;
        if (j < num_vec) {
            a = __ldg(&anchor_vec[j]);
            p = __ldg(&positive_vec[j]);
            n = __ldg(&negative_vec[j]);

            d = a.x - p.x; sum_pos += d * d;
            d = a.y - p.y; sum_pos += d * d;
            d = a.z - p.z; sum_pos += d * d;
            d = a.w - p.w; sum_pos += d * d;

            d = a.x - n.x; sum_neg += d * d;
            d = a.y - n.y; sum_neg += d * d;
            d = a.z - n.z; sum_neg += d * d;
            d = a.w - n.w; sum_neg += d * d;
        }
    }

    // Handle remaining elements if feat_size is not divisible by 4
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
    for (int off = warpSize / 2; off > 0; off /= 2) {
        sum_pos += __shfl_down_sync(warp_mask, sum_pos, off);
        sum_neg += __shfl_down_sync(warp_mask, sum_neg, off);
    }

    // Use shared memory to reduce across warps
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
        sum_pos = (tid < (blockDim.x / warpSize)) ? shared_pos[tid] : 0.f;
        sum_neg = (tid < (blockDim.x / warpSize)) ? shared_neg[tid] : 0.f;
        for (int off = warpSize / 2; off > 0; off /= 2) {
            sum_pos += __shfl_down_sync(warp_mask, sum_pos, off);
            sum_neg += __shfl_down_sync(warp_mask, sum_neg, off);
        }
        if (tid == 0) {
            float loss = sqrtf(sum_pos) - sqrtf(sum_neg) + c_margin;
            output[batch_idx] = (loss > 0.f) ? loss : 0.f;
        }
    }
}

// Launcher function that copies the margin into constant memory and launches the kernel
torch::Tensor triplet_margin_loss_cuda_constopt(
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

    // Copy the margin value to constant memory
    cudaMemcpyToSymbol(c_margin, &margin, sizeof(float));

    int threads = 256;
    triplet_margin_loss_kernel_constopt<<<batch_size, threads>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        feat_size);

    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda_constopt, "Triplet margin loss forward optimized with constant memory (CUDA)");
}
