#include <torch/extension.h>

__device__ float find_max_logit(const float* logits, int num_classes) {
    float max_val = logits[0];
    #pragma unroll 4
    for (int j = 1; j < num_classes; j++) {
        max_val = fmaxf(max_val, logits[j]);
    }
    return max_val;
}

__device__ float compute_exp_sum(const float* logits, float max_val, int num_classes) {
    float sum = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < num_classes; j++) {
        sum += expf(logits[j] - max_val);
    }
    return sum;
}

__device__ float compute_sample_loss(
    const float* sample_logits,
    int target_class,
    int num_classes
) {
    // Find max logit for numerical stability
    float max_logit = find_max_logit(sample_logits, num_classes);
    
    // Compute sum of exponentials
    float sum_exp = compute_exp_sum(sample_logits, max_logit, num_classes);
    
    // Compute final loss
    float log_sum_exp = logf(sum_exp);
    return -(sample_logits[target_class] - max_logit - log_sum_exp);
}

__global__ void cross_entropy_loss_kernel(
    const float* logits,
    const int64_t* targets,
    float* losses,
    int batch_size,
    int num_classes
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        const float* sample_logits = logits + idx * num_classes;
        int target = targets[idx];
        
        losses[idx] = compute_sample_loss(sample_logits, target, num_classes);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets)
{
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

    // Optimize block size for H100
    const int threads = 128;
    const int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward (CUDA)");
}