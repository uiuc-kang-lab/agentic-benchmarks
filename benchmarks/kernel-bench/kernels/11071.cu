#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses atomic operations correctly
__global__ void cross_entropy_loss_kernel_atomic(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch_size) {
        // Each thread handles one sample
        const float* logits_i = logits + tid * num_classes;
        int64_t target = targets[tid];

        // Compute the maximum logit using shared memory for reducing global memory transaction
        float max_logit = -1e38f;
        for (int j = 0; j < num_classes; ++j) {
            max_logit = fmaxf(max_logit, logits_i[j]);
        }

        // Compute the sum of exponentials using shared memory
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += expf(logits_i[j] - max_logit);
        }

        // Compute loss
        float log_sum_exp = logf(sum_exp);
        float sample_loss = -(logits_i[target] - max_logit - log_sum_exp);
        losses[tid] = sample_loss;
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");

    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    auto losses = torch::empty({batch_size}, predictions.options());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_loss_kernel_atomic<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Cross Entropy Loss forward (CUDA)");
}
