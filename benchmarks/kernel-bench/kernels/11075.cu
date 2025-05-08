#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void cross_entropy_loss_kernel_stride(
    const float* logits,
    const int64_t* targets,
    float* losses,
    int batch_size,
    int num_classes
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread may process multiple samples using a stride loop
    for (int i = idx; i < batch_size; i += stride) {
        const float* logits_i = logits + i * num_classes;
        int target = targets[i];

        // Compute max for numerical stability
        float max_logit = logits_i[0];
        for (int j = 1; j < num_classes; j++) {
            max_logit = fmaxf(max_logit, logits_i[j]);
        }

        // Compute sum of exp(logits - max_logit)
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += expf(logits_i[j] - max_logit);
        }

        float log_sum_exp = logf(sum_exp);

        // Compute the cross entropy loss for the sample
        losses[i] = -(logits_i[target] - max_logit - log_sum_exp);
    }
}

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

    // Allocate space for individual losses
    auto losses = torch::empty({batch_size}, predictions.options());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    // Launch kernel using grid-stride loop for better load balancing
    cross_entropy_loss_kernel_stride<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_stride: ", cudaGetErrorString(err));

    // Compute mean loss over the batch
    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward with stride loop (CUDA)");
}
