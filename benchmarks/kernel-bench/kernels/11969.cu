#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

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

    // Reduced thread count to 128 for better occupancy
    extern __shared__ float shared_mem[];
    float* sh_sum_pos = shared_mem;
    float* sh_sum_neg = shared_mem + blockDim.x;

    float sum_pos = 0.0f;
    float sum_neg = 0.0f;
    int offset = batch_idx * feat_size;

    // Process elements with stride pattern for better memory coalescing
    for (int i = threadIdx.x; i < feat_size; i += blockDim.x) {
        float a = __ldg(anchor + offset + i);
        float p = __ldg(positive + offset + i);
        float n = __ldg(negative + offset + i);
        
        float d_pos = a - p;
        float d_neg = a - n;
        sum_pos += d_pos * d_pos;
        sum_neg += d_neg * d_neg;
    }

    // Store partial sums
    sh_sum_pos[threadIdx.x] = sum_pos;
    sh_sum_neg[threadIdx.x] = sum_neg;
    __syncthreads();

    // Warp-level reduction for first 64 threads
    if (threadIdx.x < 64) {
        sh_sum_pos[threadIdx.x] += sh_sum_pos[threadIdx.x + 64];
        sh_sum_neg[threadIdx.x] += sh_sum_neg[threadIdx.x + 64];
    }
    __syncthreads();

    // First warp does final reduction
    if (threadIdx.x < 32) {
        // Warp-synchronized implicit reduction
        volatile float* vsh_pos = sh_sum_pos;
        volatile float* vsh_neg = sh_sum_neg;
        
        if (threadIdx.x < 32) vsh_pos[threadIdx.x] += vsh_pos[threadIdx.x + 32];
        if (threadIdx.x < 16) vsh_pos[threadIdx.x] += vsh_pos[threadIdx.x + 16];
        if (threadIdx.x < 8)  vsh_pos[threadIdx.x] += vsh_pos[threadIdx.x + 8];
        if (threadIdx.x < 4)  vsh_pos[threadIdx.x] += vsh_pos[threadIdx.x + 4];
        if (threadIdx.x < 2)  vsh_pos[threadIdx.x] += vsh_pos[threadIdx.x + 2];
        if (threadIdx.x < 1)  vsh_pos[threadIdx.x] += vsh_pos[threadIdx.x + 1];

        if (threadIdx.x < 32) vsh_neg[threadIdx.x] += vsh_neg[threadIdx.x + 32];
        if (threadIdx.x < 16) vsh_neg[threadIdx.x] += vsh_neg[threadIdx.x + 16];
        if (threadIdx.x < 8)  vsh_neg[threadIdx.x] += vsh_neg[threadIdx.x + 8];
        if (threadIdx.x < 4)  vsh_neg[threadIdx.x] += vsh_neg[threadIdx.x + 4];
        if (threadIdx.x < 2)  vsh_neg[threadIdx.x] += vsh_neg[threadIdx.x + 2];
        if (threadIdx.x < 1)  vsh_neg[threadIdx.x] += vsh_neg[threadIdx.x + 1];

        if (threadIdx.x == 0) {
            float total_pos = sh_sum_pos[0];
            float total_neg = sh_sum_neg[0];
            float loss = sqrtf(total_pos) - sqrtf(total_neg) + margin;
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

    // Using 256 threads per block for better occupancy, improving parallelism
    const int threads = 128;
    const int shared_mem_size = 2 * threads * sizeof(float);
    
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