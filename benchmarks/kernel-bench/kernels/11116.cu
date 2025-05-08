#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel with stride loop to handle batch sizes larger than the number of available threads
__global__ void cross_entropy_loss_stride_kernel(
    const float* logits,
    const int64_t* targets,
    float* losses,
    int batch_size,
    int num_classes
)
{
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Each thread processes multiple samples via a stride loop
    for (int i = globalThreadId; i < batch_size; i += totalThreads) {
        // Pointer to the logits for this sample
        const float* logits_i = logits + i * num_classes;
        int64_t target = targets[i];

        // Compute max logit for numerical stability using warp-level operations
        float max_logit = logits_i[0];
        #pragma unroll
        for (int j = 1; j < num_classes; j++) {
            max_logit = fmaxf(max_logit, logits_i[j]);
        }

        // Compute the sum of exponentials (logits - max_logit) for numerical stability
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += expf(logits_i[j] - max_logit);
        }

        // Log sum of exponentials
        float log_sum_exp = logf(sum_exp);
        
        // Compute loss for this sample
        // loss = - (logit_target - max_logit - log_sum_exp)
        losses[i] = - (logits_i[target] - max_logit - log_sum_exp);
    }
}

// Forward function to launch the kernel

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

    // Create an output tensor for the per-sample losses
    auto losses = torch::empty({batch_size}, predictions.options());

    // Configure CUDA kernel launch parameters
    int threads = 256;
    // Use a multiple of threads to ensure sufficient coverage
    int blocks = (batch_size + threads - 1) / threads;
    // Optionally, you can cap the number of blocks to a maximum number if desired

    cross_entropy_loss_stride_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_stride_kernel: ", cudaGetErrorString(err));

    // Return the mean loss over the batch
    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward with stride loops (CUDA)");
}
