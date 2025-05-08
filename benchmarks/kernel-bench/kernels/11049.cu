#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: each warp computes the loss for one sample using warp-level shuffle reductions
__global__ void cross_entropy_loss_kernel_warp_shfl(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    // Each warp processes one sample. Compute global warp id and lane id
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / 32;  // assume warp size is 32
    int lane = global_thread_id % 32;

    // Ensure this warp corresponds to a valid sample
    if (warp_id >= batch_size) return;

    // Pointer to the logits for this sample
    const float* sample_logits = logits + warp_id * num_classes;
    int target_class = targets[warp_id];

    // Step 1: Compute the maximum logit using warp-level reduction
    float local_max = -1e38f; // a very small number
    for (int j = lane; j < num_classes; j += 32) {
        local_max = fmaxf(local_max, sample_logits[j]);
    }

    unsigned int mask = 0xFFFFFFFF;
    // Reduce within the warp to obtain the maximum
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(mask, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    // Broadcast the maximum value to all lanes in the warp
    float max_val = __shfl_sync(mask, local_max, 0);

    // Step 2: Compute the sum of exp(logits - max_val) using warp-level reduction
    float local_sum = 0.0f;
    for (int j = lane; j < num_classes; j += 32) {
        local_sum += expf(sample_logits[j] - max_val);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    float total_sum = __shfl_sync(mask, local_sum, 0);

    // Step 3: Retrieve the logit corresponding to the target class
    float target_logit = 0.0f;
    if (lane == 0) {
        target_logit = sample_logits[target_class];
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

    // Allocate output tensor for per-sample losses
    auto losses = torch::empty({batch_size}, predictions.options());

    // Launch configuration: each warp processes one sample
    int threads_per_block = 128; // Must be a multiple of 32
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    cross_entropy_loss_kernel_warp_shfl<<<num_blocks, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_warp_shfl: ", cudaGetErrorString(err));

    // Compute mean loss over the batch and return
    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward (CUDA) with warp-level shfl reduction");
}
