#include <torch/extension.h>
#include <cmath>
#include <cfloat>

__global__ void hybrid_optimized_cross_entropy_loss_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    // Use warp-based reduction by assigning one warp per sample for reduction across classes.
    int sample = blockIdx.x * blockDim.y + threadIdx.y;
    if (sample >= batch_size) return;

    int tid = threadIdx.x;
    const int warpSize = 32;  // Warp size is 32
    float sum_exp = 0.0f;
    float max_logit = -FLT_MAX;
    const float* sample_logits = logits + sample * num_classes;

    // Use one warp (32 threads) per sample for reducing over classes
    // Phase 1: Find max
    for (int j = tid; j < num_classes; j += blockDim.x) {
        float val = sample_logits[j];
        max_logit = fmaxf(max_logit, val);
    }
    for (int stride = warpSize / 2; stride > 0; stride /= 2) {
        float val = __shfl_down_sync(0xffffffff, max_logit, stride);
        max_logit = fmaxf(max_logit, val);
    }

    // Broadcast the maximum logit inside the warp
    max_logit = __shfl_sync(0xffffffff, max_logit, 0);

    // Phase 2: Sum of exponentials
    for (int j = tid; j < num_classes; j += blockDim.x) {
        sum_exp += expf(sample_logits[j] - max_logit);
    }
    for (int stride = warpSize / 2; stride > 0; stride /= 2) {
        float val = __shfl_down_sync(0xffffffff, sum_exp, stride);
        sum_exp += val;
    }

    // Compute log_sum_exp and the final loss
    if (tid % warpSize == 0) {
        int64_t target = targets[sample];
        float log_sum_exp = logf(sum_exp);
        losses[sample] = - (sample_logits[target] - max_logit - log_sum_exp);
    }
}

// Host function

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

    auto losses = torch::empty({batch_size}, predictions.options());

    const int threads_x = 32; // Warp size of 32 ensures full warp use
    const int threads_y = 1;  // One warp per sample
    dim3 block(threads_x, threads_y);
    int grid_x = (batch_size + threads_y - 1) / threads_y;
    dim3 grid(grid_x);

    hybrid_optimized_cross_entropy_loss_kernel<<<grid, block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in hybrid_optimized_cross_entropy_loss_kernel: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid Optimized Cross Entropy Loss forward (CUDA)");
}