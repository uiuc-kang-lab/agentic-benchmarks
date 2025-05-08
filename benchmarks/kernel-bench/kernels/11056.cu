#include <torch/extension.h>

__global__ void cross_entropy_loss_kernel_optimized(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int grid_size = gridDim.x * block_size;

    // Process multiple elements per thread using grid-stride loop
    for (int i = bid * block_size + tid; i < batch_size; i += grid_size) {
        const float* logits_i = logits + i * num_classes;
        int64_t target = targets[i];

        // Calculate max logit
        float max_logit = logits_i[0];
        for (int j = 1; j < num_classes; ++j) {
            max_logit = fmaxf(max_logit, logits_i[j]);
        }

        // Calculate sum of exp(logits - max_logit)
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += expf(logits_i[j] - max_logit);
        }

        // Calculate loss
        float log_sum_exp = logf(sum_exp);
        losses[i] = -(logits_i[target] - max_logit - log_sum_exp);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");

    const int batch_size = predictions.size(0);
    const int num_classes = predictions.size(1);
    auto losses = torch::empty({batch_size}, predictions.options());

    // Configure optimal block and grid sizes
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_loss_kernel_optimized<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    
    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CrossEntropyLoss forward (CUDA)");
}
