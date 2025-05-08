/*
 * This CUDA kernel combines vectorized memory accesses (using float4) for high throughput
 * with warp-level and block-level reduction techniques to efficiently compute KL divergence.
 * It processes groups of 4 elements with vectorized loads and then handles any tail elements.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined efficient kernel using float4 vectorized loads and warp-level reductions
__global__ void efficient_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t n) {

    // Compute global thread ID and total thread count
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Process groups of 4 elements using vectorized loads
    int64_t num_vec = n / 4;  // number of complete float4 groups
    float thread_sum = 0.0f;

    // Cast pointers to float4 for vectorized access
    const float4* log_pred_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* target_vec = reinterpret_cast<const float4*>(targets);

    for (int64_t i = tid; i < num_vec; i += total_threads) {
        float4 lp = log_pred_vec[i];
        float4 tgt = target_vec[i];
        // KL divergence: exp(lp) - tgt * lp for each element
        thread_sum += expf(lp.x) - tgt.x * lp.x;
        thread_sum += expf(lp.y) - tgt.y * lp.y;
        thread_sum += expf(lp.z) - tgt.z * lp.z;
        thread_sum += expf(lp.w) - tgt.w * lp.w;
    }

    // Handle any remaining elements (tail) that don't fit into a float4
    int64_t tail_start = num_vec * 4;
    for (int64_t i = tail_start + tid; i < n; i += total_threads) {
        float lp = log_predictions[i];
        float tgt = targets[i];
        thread_sum += expf(lp) - tgt * lp;
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Shared memory for block-level reduction (one value per warp)
    __shared__ float warp_sums[32];  // supports up to 1024 threads per block (32 warps)
    int lane = threadIdx.x & 31;    // lane index within the warp
    int warpId = threadIdx.x >> 5;  // warp index within the block
    if (lane == 0) {
        warp_sums[warpId] = thread_sum;
    }
    __syncthreads();

    // Block-level reduction by thread 0
    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < num_warps; i++) {
            block_sum += warp_sums[i];
        }
        atomicAdd(output, block_sum);
    }
}

// CUDA function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int64_t n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Launch configuration: 256 threads per block
    const int threads = 256;
    // Estimate blocks to cover the input; each thread processes roughly 4 elements in the vectorized loop
    int blocks = (n + threads * 4 - 1) / (threads * 4);
    blocks = blocks > 0 ? blocks : 1;

    efficient_kl_div_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Efficient KL divergence forward (CUDA)");
}
