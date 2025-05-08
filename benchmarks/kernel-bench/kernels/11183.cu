// Combined CUDA kernel for Smooth L1 Loss with vectorized loads and warp-level reduction
// This implementation fuses the efficient vectorized memory accesses (using float4) with warp-level reduction using shuffle intrinsics.
// It processes the bulk of the data in 4-element chunks and handles any remainder with scalar loads.
// The warp-level reduction minimizes shared memory usage and synchronizations, while a final shared reduction aggregates results per block.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Combined kernel: Uses vectorized float4 loads and warp-level reduction to compute smooth L1 loss
__global__ void smooth_l1_loss_kernel_combined(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Vectorized processing: process groups of 4 elements at a time
    int vec_count = n_elements / 4;
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);
    for (int i = idx; i < vec_count; i += stride) {
        float4 p = __ldg(pred4 + i);
        float4 t = __ldg(targ4 + i);

        float diff = p.x - t.x;
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        diff = p.y - t.y;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        diff = p.z - t.z;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        diff = p.w - t.w;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Process remaining elements that don't fit into a float4 vector
    int remainder_start = vec_count * 4;
    for (int i = remainder_start + idx; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Perform warp-level reduction using shuffle intrinsics
    thread_sum = warp_reduce_sum(thread_sum);

    // Each warp's lane 0 writes its partial sum to shared memory
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;
    __shared__ float warp_sums[32]; // Supports up to 1024 threads per block (32 warps)
    if (lane == 0) {
        warp_sums[warpId] = thread_sum;
    }
    __syncthreads();

    // Final reduction by the first warp
    float block_sum = 0.0f;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        block_sum = warp_sums[lane];
    }
    if (threadIdx.x < warpSize) {
        block_sum = warp_reduce_sum(block_sum);
    }

    // The first thread atomically adds the block's contribution to the global output (average over n_elements)
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum / n_elements);
    }
}

// Host function wrapping the combined kernel
torch::Tensor smooth_l1_loss_combined(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input shape mismatch");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Non-contiguous inputs");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    int n_elements = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    int grid_size = (n_elements / 4 + block_size - 1) / block_size;
    grid_size = grid_size > 0 ? grid_size : 1;

    smooth_l1_loss_kernel_combined<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_combined, "Combined Optimized Smooth L1 Loss with vectorized loads and warp-level reduction");
}
