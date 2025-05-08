#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel uses one warp (32 threads) per sample to compute the cross entropy loss
// using warp-level primitives for reduction, thereby avoiding unnecessary __syncthreads().
__global__ void ce_loss_warp_kernel(
    const float* logits,
    const int64_t* targets,
    float* losses,
    int num_classes
)
{
    // Each block handles one sample. Ensure blockDim.x == 32.
    int sample = blockIdx.x;
    int lane = threadIdx.x; // lane should be in [0,31]

    // Pointer to logits for this sample
    const float* sample_logits = logits + sample * num_classes;
    int64_t target = targets[sample];

    // Compute the maximum logit for numerical stability.
    float local_max = -INFINITY;
    for (int j = lane; j < num_classes; j += 32) {
        float val = sample_logits[j];
        local_max = fmaxf(local_max, val);
    }
    // Warp-level reduction to compute maximum without __syncthreads()
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    // Broadcast the maximum value to all lanes in the warp
    float max_val = __shfl_sync(0xffffffff, local_max, 0);

    // Compute the sum of exponentials of (logit - max_val) in parallel
    float local_sum = 0.0f;
    for (int j = lane; j < num_classes; j += 32) {
        local_sum += expf(sample_logits[j] - max_val);
    }
    // Warp-level reduction to sum up values
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    float sum_exp = __shfl_sync(0xffffffff, local_sum, 0);

    // Lane 0 computes the final loss and writes to global memory
    if (lane == 0) {
        float log_sum_exp = logf(sum_exp);
        float loss = - (sample_logits[target] - max_val - log_sum_exp);
        losses[sample] = loss;
    }
}


// The forward function launches one block per sample, with exactly one warp (32 threads) per block.
// This minimizes synchronization overhead by using warp-level shuffles only.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    auto losses = torch::empty({batch_size}, predictions.options());

    // Launch one block per sample with 32 threads (one warp per sample)
    int threads = 32;
    int blocks = batch_size;
    ce_loss_warp_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in ce_loss_warp_kernel: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward using warp-level reduction (CUDA)");
}
