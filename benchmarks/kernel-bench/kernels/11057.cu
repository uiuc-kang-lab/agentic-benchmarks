#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel using grid-stride loops for boundary handling
__global__ void cross_entropy_loss_kernel_stride(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    const int batch_size,
    const int num_classes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple samples using a stride loop
    for (int i = tid; i < batch_size; i += stride) {
        const float* logits_i = logits + i * num_classes;
        int64_t target = targets[i];

        // Compute the maximum logit for numerical stability
        float max_logit = logits_i[0];
        for (int j = 1; j < num_classes; ++j) {
            max_logit = fmaxf(max_logit, logits_i[j]);
        }

        // Compute the sum of exponentials with numerical stabilization
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += expf(logits_i[j] - max_logit);
        }

        // Compute the final loss for the sample
        losses[i] = -(logits_i[target] - max_logit - logf(sum_exp));
    }
}

// Forward function integrating the CUDA kernel
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");

    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");

    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    const int batch_size = predictions.size(0);
    const int num_classes = predictions.size(1);

    auto losses = torch::empty({batch_size}, predictions.options());

    // Launch configuration using grid-stride loops to cover the entire batch
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_loss_kernel_stride<<<blocks, threads>>>(
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
    m.def("forward", &forward, "CrossEntropyLoss forward (CUDA stride loop)");
}
