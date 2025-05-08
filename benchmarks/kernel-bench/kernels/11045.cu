#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cmath>

// Combined Kernel: uses a 2D block mapping (samples in y-dim, reduction over classes in x-dim) 
// and leverages warp-level primitives (__shfl_down_sync) to perform efficient intra-warp reductions for both max and sum calculations.
// This reduces the number of synchronizations and shared memory traffic compared to a full shared-memory reduction.

__global__ void cross_entropy_loss_kernel_combined(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    // Each block processes several samples along the y-dimension
    int sample = blockIdx.x * blockDim.y + threadIdx.y;
    if (sample >= batch_size) return;

    // Pointer to the logits for this sample
    const float* sample_logits = logits + sample * num_classes;

    // Use 2D block mapping: threadIdx.x will handle portions of the class dimension
    // and we use warp-level reduction to efficiently reduce across threads.
    const int warpSize = 32;
    int lane = threadIdx.x % warpSize;      // lane index within a warp
    int warpId = threadIdx.x / warpSize;      // warp index in the block (for this sample row)
    int nWarps = (blockDim.x + warpSize - 1) / warpSize;  // number of warps per sample

    // Declare shared memory. Each sample uses a contiguous region of nWarps floats.
    // This region will be used to store warp-level partial results for both max and sum reductions.
    extern __shared__ float sdata[];  // Size: blockDim.y * nWarps * sizeof(float)

    // --------------------- Phase 1: Compute Maximum Logit ---------------------
    float local_max = -FLT_MAX;
    // Each thread processes a subset of classes
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        float val = sample_logits[j];
        local_max = fmaxf(local_max, val);
    }

    // Intra-warp reduction for maximum using warp shuffle
    unsigned int mask = 0xFFFFFFFF;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(mask, local_max, offset);
        local_max = fmaxf(local_max, other);
    }

    // Each warp's lane 0 writes its maximum to shared memory
    if (lane == 0) {
        sdata[threadIdx.y * nWarps + warpId] = local_max;
    }
    __syncthreads();

    // Inter-warp reduction: first nWarps threads in the sample row reduce the partial max values
    float max_val;
    if (threadIdx.x < nWarps) {
        float val = sdata[threadIdx.y * nWarps + threadIdx.x];
        // Use a simple loop reduction (nWarps is typically small)
        for (int offset = 1; offset < nWarps; offset *= 2) {
            float other = __shfl_down_sync(mask, val, offset);
            val = fmaxf(val, other);
        }
        if (threadIdx.x == 0) {
            max_val = val;
            // Store the final maximum for later broadcast
            sdata[threadIdx.y * nWarps] = max_val;
        }
    }
    __syncthreads();
    max_val = sdata[threadIdx.y * nWarps];

    // --------------------- Phase 2: Compute Sum of Exp(logits - max) ---------------------
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        local_sum += expf(sample_logits[j] - max_val);
    }
    
    // Intra-warp reduction for sum using warp shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Each warp's lane 0 writes its partial sum to shared memory
    if (lane == 0) {
        sdata[threadIdx.y * nWarps + warpId] = local_sum;
    }
    __syncthreads();

    // Inter-warp reduction for sum
    float sum_exp;
    if (threadIdx.x < nWarps) {
        float val = sdata[threadIdx.y * nWarps + threadIdx.x];
        for (int offset = 1; offset < nWarps; offset *= 2) {
            float other = __shfl_down_sync(mask, val, offset);
            val += other;
        }
        if (threadIdx.x == 0) {
            sum_exp = val;
            sdata[threadIdx.y * nWarps] = sum_exp;  // store final sum for broadcast
        }
    }
    __syncthreads();
    sum_exp = sdata[threadIdx.y * nWarps];

    // --------------------- Compute Final Loss ---------------------
    // Only one thread per sample (e.g., thread with threadIdx.x == 0) writes the loss
    if (threadIdx.x == 0) {
        int64_t target = targets[sample];
        float loss = - (sample_logits[target] - max_val - logf(sum_exp));
        losses[sample] = loss;
    }
}

// Host function: validates inputs, sets kernel launch parameters, and calls the combined CUDA kernel

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    // Input validations
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be an Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    // Allocate output tensor for losses
    auto losses = torch::empty({batch_size}, predictions.options());

    // Define 2D block dimensions: x dimension for reduction over classes, y dimension to process multiple samples
    const int threads_x = 128;
    const int threads_y = 4;  // Number of samples processed per block
    dim3 block(threads_x, threads_y);
    int grid_x = (batch_size + threads_y - 1) / threads_y;
    dim3 grid(grid_x);

    // Shared memory: each sample uses nWarps floats, where nWarps = ceil(threads_x / 32)
    int nWarps = (threads_x + 31) / 32;
    size_t shared_mem_size = threads_y * nWarps * sizeof(float);

    // Launch the combined kernel
    cross_entropy_loss_kernel_combined<<<grid, block, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_combined: ", cudaGetErrorString(err));

    // Return mean loss over the batch
    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined Cross Entropy Loss forward (CUDA) using warp-level reduction");
}
