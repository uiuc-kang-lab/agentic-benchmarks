#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float compute_max_logit(const float* logits, int num_classes) {
    float max_val = logits[0];
    for (int j = 1; j < num_classes; j++) {
        max_val = fmaxf(max_val, logits[j]);
    }
    return max_val;
}

__device__ float compute_sum_exp_diff(const float* logits, float max_val, int num_classes) {
    float sum = 0.0f;
    for (int j = 0; j < num_classes; j++) {
        sum += expf(logits[j] - max_val);
    }
    return sum;
}

__device__ float compute_loss(float logit_target, float max_logit, float log_sum_exp) {
    return -(logit_target - max_logit - log_sum_exp);
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
    int totalThreads = gridDim.x * blockDim.x;

    for (int i = idx; i < batch_size; i += totalThreads) {
        const float* sample_logits = logits + i * num_classes;
        int64_t target = targets[i];

        // Maximum logit for the sample
        float max_logit = compute_max_logit(sample_logits, num_classes);
        
        // Exponential sum difference
        float sum_exp = compute_sum_exp_diff(sample_logits, max_logit, num_classes);

        // Compute log(sum(exp(...)))
        float log_sum_exp = logf(sum_exp);

        // Compute final loss for current sample
        losses[i] = compute_loss(sample_logits[target], max_logit, log_sum_exp);
    }
}

// Forward function to launch the kernel

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

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

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
    m.def("forward", &forward, "Cross Entropy Loss forward with modular functions (CUDA)");
}
