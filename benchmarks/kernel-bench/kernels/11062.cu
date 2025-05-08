#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

// Optimized CUDA kernel for cross entropy loss using warp-level shuffle reductions
// Each warp processes one sample. We use __restrict__ pointers and lane-stride loops to
// efficiently compute the maximum logit and the sum of exponentials for numerical stability.

__global__ void cross_entropy_loss_kernel_opt(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    // Compute global thread id and derive warp id and lane id (warp size assumed to be 32)
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= batch_size) return;

    // Pointer to the logits for the sample corresponding to this warp
    const float* sample_logits = logits + warp_id * num_classes;
    int target_class = targets[warp_id];

    // First pass: Compute the maximum logit for numerical stability using lane-stride loop
    float local_max = -FLT_MAX;
    for (int j = lane; j < num_classes; j += 32) {
        float val = sample_logits[j];
        local_max = fmaxf(local_max, val);
    }
    
    // Warp-level reduction to obtain the maximum value (using warp shuffle intrinsics)
    unsigned int mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(mask, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    // Broadcast the maximum value to all lanes in the warp
    float max_val = __shfl_sync(mask, local_max, 0);

    // Second pass: Compute the sum of exp(logits - max_val) using lane-stride loop
    float local_sum = 0.0f;
    for (int j = lane; j < num_classes; j += 32) {
        local_sum += expf(sample_logits[j] - max_val);
    }
    
    // Warp-level reduction to sum up the partial exponentials
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    float total_sum = __shfl_sync(mask, local_sum, 0);

    // Retrieve target logit: only one thread per warp loads it and then broadcasts
    float target_logit = 0.0f;
    if (lane == 0) {
        target_logit = sample_logits[target_class];
    }
    target_logit = __shfl_sync(mask, target_logit, 0);

    // Compute the cross entropy loss for this sample
    float loss = -(target_logit - max_val - logf(total_sum));

    // Only lane 0 writes the result for this sample
    if (lane == 0) {
        losses[warp_id] = loss;
    }
}

// Forward function wrapping the optimized kernel call
// Applies necessary checks on tensor dimensions and types

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

    // Allocate tensor for per-sample losses
    auto losses = torch::empty({batch_size}, predictions.options());

    // Launch configuration: each warp (32 threads) processes one sample
    int threads_per_block = 128; // Must be a multiple of 32
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    cross_entropy_loss_kernel_opt<<<num_blocks, threads_per_block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_opt: ", cudaGetErrorString(err));

    // Compute mean loss over the batch
    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Cross Entropy Loss forward (CUDA) using warp-level shfl reductions");
}
