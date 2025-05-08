#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function to compute the maximum logit for numerical stability
__device__ __forceinline__ float device_max_logit(const float* logits, int num_classes) {
    float max_val = logits[0];
    for (int j = 1; j < num_classes; ++j) {
        max_val = fmaxf(max_val, logits[j]);
    }
    return max_val;
}

// Device function to compute the sum of exponentials of logits shifted by max_val
__device__ __forceinline__ float device_sum_exp(const float* logits, int num_classes, float max_val) {
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; ++j) {
        sum_exp += expf(logits[j] - max_val);
    }
    return sum_exp;
}

// Device function to compute the cross entropy loss for a single sample
__device__ __forceinline__ float compute_cross_entropy_loss(const float* logits, int64_t target, int num_classes) {
    float max_val = device_max_logit(logits, num_classes);
    float sum_exp = device_sum_exp(logits, num_classes, max_val);
    // Compute loss: - (logit[target] - max_val - log(sum_exp))
    return -(logits[target] - max_val - logf(sum_exp));
}

// CUDA kernel using modular device functions
__global__ void cross_entropy_loss_kernel_modular(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        // Process one sample per thread
        const float* logits_sample = logits + i * num_classes;
        int64_t target = targets[i];
        losses[i] = compute_cross_entropy_loss(logits_sample, target, num_classes);
    }
}

// Forward function exposed to PyTorch
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be an Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    auto losses = torch::empty({batch_size}, predictions.options());
    
    // Launch parameters
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_loss_kernel_modular<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in kernel: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward (CUDA) with modular device functions");
}
