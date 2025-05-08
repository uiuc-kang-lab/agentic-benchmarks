#include <torch/extension.h>

__global__ void cross_entropy_loss_kernel(
    const float* logits,
    const int64_t* targets,
    float* losses,
    int batch_size,
    int num_classes
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size)
    {
        // Get pointer to logits for sample i
        const float* logits_i = logits + i * num_classes;
        int64_t target = targets[i];

        // Compute max logit for numerical stability
        float max_logit = logits_i[0];
        for (int j = 1; j < num_classes; j++)
        {
            if (logits_i[j] > max_logit)
                max_logit = logits_i[j];
        }

        // Compute sum of exp(logits - max_logit)
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++)
        {
            sum_exp += expf(logits_i[j] - max_logit);
        }

        // Compute log_sum_exp
        float log_sum_exp = logf(sum_exp);

        // Compute loss for this sample
        float loss = - (logits_i[target] - max_logit - log_sum_exp);
        losses[i] = loss;
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets)
{
    // Ensure inputs are on CUDA
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");

    // Ensure inputs have correct dimensions
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");

    // Ensure data types are correct
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    // Output tensor for losses per sample
    auto losses = torch::empty({batch_size}, predictions.options());

    // Launch CUDA kernel
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel: ", cudaGetErrorString(err));

    // Compute mean loss over batch
    auto loss = losses.mean();

    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward (CUDA)");
}