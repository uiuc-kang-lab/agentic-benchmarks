#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized kernel for Triplet Margin Loss using __ldg() for read-only load and 128-bit aligned vectorized loads (float4).
// Each block handles one batch element and uses shared memory reduction for performance.

__global__ void triplet_margin_loss_kernel_optimized(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Allocate shared memory for reduction: two arrays for positive and negative sums
    extern __shared__ float shared_mem[];
    float* sh_sum_pos = shared_mem;
    float* sh_sum_neg = shared_mem + blockDim.x;

    float sum_pos = 0.0f;
    float sum_neg = 0.0f;
    int offset = batch_idx * feat_size;

    // Use vectorized loads if possible by processing 4 floats (128-bit) at a time
    int vectorized_end = (feat_size / 4) * 4;
    const float4* anchor_vec = reinterpret_cast<const float4*>(anchor + offset);
    const float4* positive_vec = reinterpret_cast<const float4*>(positive + offset);
    const float4* negative_vec = reinterpret_cast<const float4*>(negative + offset);
    int num_vec = vectorized_end / 4;

    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 a4 = __ldg(&anchor_vec[i]);
        float4 p4 = __ldg(&positive_vec[i]);
        float4 n4 = __ldg(&negative_vec[i]);

        // Compute squared differences for anchor-positive
        float d0 = a4.x - p4.x;
        float d1 = a4.y - p4.y;
        float d2 = a4.z - p4.z;
        float d3 = a4.w - p4.w;
        sum_pos += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;

        // Compute squared differences for anchor-negative
        d0 = a4.x - n4.x;
        d1 = a4.y - n4.y;
        d2 = a4.z - n4.z;
        d3 = a4.w - n4.w;
        sum_neg += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    // Process any remaining elements that don't fit into a float4
    for (int i = vectorized_end + threadIdx.x; i < feat_size; i += blockDim.x) {
        float a = __ldg(anchor + offset + i);
        float p = __ldg(positive + offset + i);
        float n = __ldg(negative + offset + i);
        float d_pos = a - p;
        float d_neg = a - n;
        sum_pos += d_pos * d_pos;
        sum_neg += d_neg * d_neg;
    }

    // Store partial sums in shared memory
    sh_sum_pos[threadIdx.x] = sum_pos;
    sh_sum_neg[threadIdx.x] = sum_neg;
    __syncthreads();

    // Perform reduction within the block to get total sum for positive and negative distances
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_sum_pos[threadIdx.x] += sh_sum_pos[threadIdx.x + s];
            sh_sum_neg[threadIdx.x] += sh_sum_neg[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float total_pos = sh_sum_pos[0];
        float total_neg = sh_sum_neg[0];
        // Compute final loss: max(0, sqrt(total_pos) - sqrt(total_neg) + margin)
        float loss = sqrtf(total_pos) - sqrtf(total_neg) + margin;
        output[batch_idx] = (loss > 0.0f) ? loss : 0.0f;
    }
}

// CUDA launcher for the optimized kernel
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

    int threads = 256;
    // Allocate shared memory: 2 arrays of 'threads' floats (for positive and negative sums)
    int shared_mem_size = 2 * threads * sizeof(float);
    
    // Launch one block per batch element
    triplet_margin_loss_kernel_optimized<<<batch_size, threads, shared_mem_size>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        output.data_ptr<float>(),
        margin,
        batch_size,
        feat_size);
    
    // Return the mean loss over the batch
    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cuda_optimized, "Triplet margin loss forward optimized (CUDA)");
}
