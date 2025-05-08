#include <torch/extension.h>

constexpr int WARP_SIZE = 32;

__global__ void cross_entropy_loss_kernel(
    const float* logits,
    const int64_t* targets,
    float* losses,
    int batch_size,
    int num_classes
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    const float* logits_i = logits + i * num_classes;
    const int target = targets[i];

    // Warp-aligned max reduction
    float max_logit = -FLT_MAX;
    for (int j = threadIdx.x % WARP_SIZE; j < num_classes; j += WARP_SIZE) {
        max_logit = fmaxf(max_logit, logits_i[j]);
    }
    
    // Warp-wide max reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        max_logit = fmaxf(max_logit, __shfl_xor_sync(0xffffffff, max_logit, offset));
    }

    // Warp-aligned exp sum calculation
    float sum_exp = 0.0f;
    for (int j = threadIdx.x % WARP_SIZE; j < num_classes; j += WARP_SIZE) {
        sum_exp += expf(logits_i[j] - max_logit);
    }

    // Warp-wide sum reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
    }

    // Only first thread in warp computes final loss
    if ((threadIdx.x % WARP_SIZE) == 0) {
        const float log_sum_exp = logf(sum_exp);
        losses[i] = -(logits_i[target] - max_logit - log_sum_exp);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(predictions.dim() == 2 && targets.dim() == 1, "Invalid input dimensions");

    const int batch_size = predictions.size(0);
    const int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "Batch size mismatch");

    auto losses = torch::empty({batch_size}, predictions.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel error: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized Cross Entropy Loss");
}