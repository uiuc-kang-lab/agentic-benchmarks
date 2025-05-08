#include <torch/extension.h>

__forceinline__ __device__ float compute_max_logit(const float* logits, int num_classes) {
    float max_val = logits[0];
    #pragma unroll 4
    for (int j = 1; j < num_classes; j++) {
        max_val = fmaxf(max_val, logits[j]);
    }
    return max_val;
}

__forceinline__ __device__ float compute_exp_sum(float max_val, const float* logits, int num_classes) {
    float sum = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < num_classes; j++) {
        sum += __expf(logits[j] - max_val);
    }
    return sum;
}

__global__ void cross_entropy_loss_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = idx; i < batch_size; i += total_threads) {
        const float* sample_logits = logits + i * num_classes;
        int target = targets[i];
        
        float max_logit = compute_max_logit(sample_logits, num_classes);
        float sum_exp = compute_exp_sum(max_logit, sample_logits, num_classes);
        float log_sum_exp = logf(sum_exp);
        
        losses[i] = -(sample_logits[target] - max_logit - log_sum_exp);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    auto losses = torch::empty({batch_size}, predictions.options());

    // H100-optimized parameters
    const int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    blocks = min(blocks, 128);  // Cap blocks for better occupancy

    cross_entropy_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error:", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CE loss with grid stride and unrolling");
}
