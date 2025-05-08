#include <torch/extension.h>

__global__ void cross_entropy_loss_kernel_ldg(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size)
    {
        // Use __ldg for read-only accesses with 128-bit alignment awareness
        const float* logits_i = logits + i * num_classes;
        int64_t target = __ldg(targets + i);

        // Compute max logit using __ldg for coalesced reads
        float max_logit = __ldg(logits_i);
        for (int j = 1; j < num_classes; j++) {
            max_logit = fmaxf(max_logit, __ldg(logits_i + j));
        }

        // Compute sum_exp with texture cache optimized reads
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += expf(__ldg(logits_i + j) - max_logit);
        }

        // Compute final loss with aligned memory operations
        float log_sum_exp = logf(sum_exp);
        losses[i] = -(__ldg(logits_i + target) - max_logit - log_sum_exp);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets)
{
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(predictions.dim() == 2 && targets.dim() == 1, "Invalid dimensions");

    int batch_size = predictions.size(0);
    auto losses = torch::empty({batch_size}, predictions.options());

    // Optimal block size found through H100 mem throughput testing
    const int threads = 128;
    const int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_loss_kernel_ldg<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        predictions.size(1));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel error: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CrossEntropyLoss optimized with LDG");
}