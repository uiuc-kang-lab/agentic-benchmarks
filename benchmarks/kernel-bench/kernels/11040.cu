#include <torch/extension.h>

__global__ void cross_entropy_loss_kernel_warp_optimized(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_size = 32;
    int lane_id = threadIdx.x % warp_size;

    for (int i = tid / warp_size; i < batch_size; i += (gridDim.x * blockDim.x) / warp_size)
    {
        // Get pointer to logits for sample i
        const float* logits_i = logits + i * num_classes;
        int64_t target = targets[i];

        // Compute max logit with warp reduction
        float max_logit = -FLT_MAX;
        for (int j = lane_id; j < num_classes; j += warp_size)
        {
            max_logit = fmaxf(max_logit, logits_i[j]);
        }
        #pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset /= 2)
        {
            max_logit = fmaxf(max_logit, __shfl_down_sync(0xFFFFFFFF, max_logit, offset));
        }

        // Broadcast max logit to all lanes
        max_logit = __shfl_sync(0xFFFFFFFF, max_logit, 0);

        // Compute sum of exp(logits - max_logit)
        float sum_exp = 0.0f;
        for (int j = lane_id; j < num_classes; j += warp_size)
        {
            sum_exp += expf(logits_i[j] - max_logit);
        }
        #pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset /= 2)
        {
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
        }

        // Broadcast sum_exp to all lanes
        sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp, 0);

        if (lane_id == 0)
        {
            // Compute log_sum_exp
            float log_sum_exp = logf(sum_exp);

            // Compute loss for this sample
            float loss = -(logits_i[target] - max_logit - log_sum_exp);
            losses[i] = loss;
        }
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

    // Launch CUDA kernel with warp-level parallelism
    int threads = 256;
    int blocks = (batch_size + (threads / 32) - 1) / (threads / 32);

    cross_entropy_loss_kernel_warp_optimized<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_warp_optimized: ", cudaGetErrorString(err));

    // Compute mean loss over batch
    auto loss = losses.mean();

    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward (CUDA)");
}