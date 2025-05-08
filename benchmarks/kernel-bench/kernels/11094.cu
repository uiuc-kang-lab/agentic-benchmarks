#include <torch/extension.h>

__global__ void find_max_logits_kernel(
    const float* __restrict__ logits,
    float* __restrict__ max_logits,
    int batch_size,
    int num_classes
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size)
    {
        const float* logits_i = logits + i * num_classes;
        float max_logit = logits_i[0];
        
        #pragma unroll
        for (int j = 1; j < num_classes; j++)
        {
            max_logit = fmaxf(max_logit, logits_i[j]);
        }
        max_logits[i] = max_logit;
    }
}

__global__ void compute_exp_sums_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ max_logits,
    float* __restrict__ sum_exps,
    int batch_size,
    int num_classes
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size)
    {
        const float* logits_i = logits + i * num_classes;
        const float max_logit = max_logits[i];
        float sum_exp = 0.0f;
        
        #pragma unroll
        for (int j = 0; j < num_classes; j++)
        {
            sum_exp += __expf(logits_i[j] - max_logit);
        }
        sum_exps[i] = sum_exp;
    }
}

__global__ void compute_final_loss_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ max_logits,
    const float* __restrict__ sum_exps,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size)
    {
        const float* logits_i = logits + i * num_classes;
        const int64_t target = targets[i];
        const float max_logit = max_logits[i];
        const float sum_exp = sum_exps[i];
        
        float log_sum_exp = __logf(sum_exp);
        float loss = -(logits_i[target] - max_logit - log_sum_exp);
        losses[i] = loss;
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
    int threads = 512;
    int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward (CUDA)");
}