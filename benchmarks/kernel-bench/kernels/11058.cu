#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel using grid-stride loop, allowing dynamic block size selection
__global__ void cross_entropy_loss_kernel_experiment(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < batch_size; i += stride) {
        const float* logits_i = logits + i * num_classes;
        int64_t target = targets[i];

        // Compute maximum logit for numerical stability
        float max_logit = logits_i[0];
        for (int j = 1; j < num_classes; ++j) {
            max_logit = fmaxf(max_logit, logits_i[j]);
        }

        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += expf(logits_i[j] - max_logit);
        }

        // Compute cross entropy loss for the sample
        losses[i] = -(logits_i[target] - max_logit - logf(sum_exp));
    }
}

// Forward function that selects an optimal block size from candidate values
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be an Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    auto losses = torch::empty({batch_size}, predictions.options());

    // Heuristic for block size: experiment with different candidates
    int block_size;
    if (batch_size < 32)
        block_size = 32;
    else if (batch_size < 64)
        block_size = 64;
    else if (batch_size < 128)
        block_size = 128;
    else if (batch_size < 256)
        block_size = 256;
    else if (batch_size < 512)
        block_size = 512;
    else
        block_size = 256; // Default value for large batch sizes

    int blocks = (batch_size + block_size - 1) / block_size;

    cross_entropy_loss_kernel_experiment<<<blocks, block_size>>>(
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
    m.def("forward", &forward, "Cross Entropy Loss forward (CUDA) with optimal blocksize experimentation");
}
