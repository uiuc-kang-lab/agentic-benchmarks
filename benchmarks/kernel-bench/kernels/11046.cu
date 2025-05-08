#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Warp-level reduction functions using shuffle intrinsics
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// This kernel fuses the 2D grid approach with warp-level reductions for both max and sum computations.
// Each block processes several samples along its y-dimension, while threads in the x-dimension cooperate
// to reduce over the num_classes dimension using efficient warp shuffle intrinsics, minimizing shared memory overhead.

__global__ void cross_entropy_loss_kernel_fused(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    // Each block processes blockDim.y samples (each sample is handled by one row of threads)
    int sample = blockIdx.x * blockDim.y + threadIdx.y;
    if (sample >= batch_size) return;

    const float* sample_logits = logits + sample * num_classes;

    // Phase 1: Compute the maximum logit value for numerical stability.
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        local_max = fmaxf(local_max, sample_logits[j]);
    }
    // Intra-warp reduction using shuffle
    local_max = warpReduceMax(local_max);

    // Determine warp id and lane within the warp
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    // Allocate shared memory for inter-warp reduction.
    // We partition the shared memory into two arrays:
    // smax: for reducing maximum values, ssum: for reducing exponential sums.
    // Size per sample row: (number of warps per row).
    extern __shared__ float sdata[];  // Total shared mem: blockDim.y * num_warps * 2 * sizeof(float)
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    float* smax = sdata;              // size: blockDim.y * num_warps
    float* ssum = smax + blockDim.y * num_warps; // size: blockDim.y * num_warps

    // Each warp leader writes its reduced maximum into shared memory.
    if (lane == 0) {
        smax[threadIdx.y * num_warps + warp_id] = local_max;
    }
    __syncthreads();

    float final_max;
    // Let threadIdx.x==0 in each sample row complete the inter-warp reduction for max.
    if (threadIdx.x == 0) {
        final_max = -FLT_MAX;
        for (int i = 0; i < num_warps; i++) {
            final_max = fmaxf(final_max, smax[threadIdx.y * num_warps + i]);
        }
        // Store the final maximum so all threads in the row can access it.
        smax[threadIdx.y * num_warps] = final_max;
    }
    __syncthreads();
    final_max = smax[threadIdx.y * num_warps];

    // Phase 2: Compute the sum of exponentials using the computed maximum.
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        local_sum += expf(sample_logits[j] - final_max);
    }
    local_sum = warpReduceSum(local_sum);

    // Each warp leader writes its partial sum into shared memory.
    if (lane == 0) {
        ssum[threadIdx.y * num_warps + warp_id] = local_sum;
    }
    __syncthreads();

    float final_sum = 0.0f;
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_warps; i++) {
            final_sum += ssum[threadIdx.y * num_warps + i];
        }
        // Compute and write the cross entropy loss for the sample.
        int64_t target = targets[sample];
        float loss = - (sample_logits[target] - final_max - logf(final_sum));
        losses[sample] = loss;
    }
}

// Host launcher that sets up the 2D grid and shared memory configuration.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be an Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    // Allocate output tensor for per-sample losses.
    auto losses = torch::empty({batch_size}, predictions.options());

    // Define 2D block dimensions: threads_x for class reduction, threads_y for processing multiple samples per block.
    const int threads_x = 128;  // Should be a multiple of warp size
    const int threads_y = 4;    // Number of samples per block
    dim3 block(threads_x, threads_y);
    int grid_x = (batch_size + threads_y - 1) / threads_y;
    dim3 grid(grid_x);

    // Shared memory: two arrays each of size (threads_y * num_warps) floats
    int num_warps = (threads_x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shared_mem_size = threads_y * num_warps * 2 * sizeof(float);

    cross_entropy_loss_kernel_fused<<<grid, block, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_fused: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Cross Entropy Loss forward (CUDA) with warp-level reduction");
}
