#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized Kernel using shared memory and warp-level shuffle reductions
__global__ void optimized_cross_entropy_loss_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    extern __shared__ float shared_logits[];

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / 32;  // assume warp size is 32
    int lane = global_thread_id % 32;

    if (warp_id >= batch_size) return;

    const float* sample_logits = logits + warp_id * num_classes;
    int target_class = targets[warp_id];

    // Load logits into shared memory
    for (int j = lane; j < num_classes; j += 32) {
        shared_logits[threadIdx.x + j - lane] = sample_logits[j];
    }
    __syncthreads();

    // Step 1: Compute the maximum logit using warp-level reduction
    float local_max = -1e38f;
    for (int j = lane; j < num_classes; j += 32) {
        local_max = fmaxf(local_max, shared_logits[threadIdx.x + j - lane]);
    }

    unsigned int mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(mask, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    float max_val = __shfl_sync(mask, local_max, 0);

    // Step 2: Compute the sum of exp(logits - max_val) using warp-level reduction
    float local_sum = 0.0f;
    for (int j = lane; j < num_classes; j += 32) {
        local_sum += expf(shared_logits[threadIdx.x + j - lane] - max_val);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    float total_sum = __shfl_sync(mask, local_sum, 0);

    // Step 3: Retrieve the logit corresponding to the target class
    float target_logit = 0.0f;
    if (lane == 0) {
        target_logit = shared_logits[threadIdx.x + target_class - lane];
    }
    target_logit = __shfl_sync(mask, target_logit, 0);

    // Step 4: Compute the loss for this sample. Only lane 0 writes the result.
    if (lane == 0) {
        losses[warp_id] = -(target_logit - max_val - logf(total_sum));
    }
}

// Forward function that wraps the kernel call
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    TORCH_CHECK(targets.size(0) == batch_size, "targets must have the same batch size as predictions");

    auto losses = torch::empty({batch_size}, predictions.options());

    int threads_per_block = 128; // Must be a multiple of 32
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    size_t shared_memory_size = threads_per_block * sizeof(float);

    optimized_cross_entropy_loss_kernel<<<num_blocks, threads_per_block, shared_memory_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in optimized_cross_entropy_loss_kernel: ", cudaGetErrorString(err));

    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Cross Entropy Loss forward (CUDA)");
}