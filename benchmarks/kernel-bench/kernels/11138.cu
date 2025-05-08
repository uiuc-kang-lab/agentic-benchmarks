#include <torch/extension.h>

__global__ void cross_entropy_loss_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    
    for (int i = idx; i < batch_size; i += total_threads) {
        const float* row = logits + i * num_classes;
        const int target = targets[i];

        // Max logit computation with unrolling
        float max_val = row[0];
        #pragma unroll 4
        for (int j = 1; j < num_classes; j++) {
            max_val = fmaxf(max_val, row[j]);
        }

        // Sum exp computation with unrolling
        float sum_exp = 0.0f;
        #pragma unroll 4
        for (int j = 0; j < num_classes; j++) {
            sum_exp += expf(row[j] - max_val);
        }

        float log_sum_exp = logf(sum_exp);
        losses[i] = -(row[target] - max_val - log_sum_exp);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(predictions.dim() == 2, "Predictions must be 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "Targets must be 1D tensor");

    const int batch_size = predictions.size(0);
    const int num_classes = predictions.size(1);
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
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CrossEntropyLoss with loop unrolling");
}