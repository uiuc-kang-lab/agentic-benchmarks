#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Define maximum number of elements allowed in constant memory (16K floats = 64KB)
#define MAX_CONSTANT_ELEMENTS 16384

// Declare constant memory for predictions (logits)
__constant__ float const_logits[MAX_CONSTANT_ELEMENTS];

// CUDA kernel that reads predictions from constant memory
__global__ void cross_entropy_loss_kernel_const(
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        int offset = i * num_classes;
        // Compute max logit for numerical stability
        float max_logit = const_logits[offset];
        for (int j = 1; j < num_classes; j++) {
            float val = const_logits[offset + j];
            max_logit = fmaxf(max_logit, val);
        }
        
        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += expf(const_logits[offset + j] - max_logit);
        }
        float log_sum_exp = logf(sum_exp);
        
        // Compute loss for the sample using the target index
        int64_t target = targets[i];
        float loss = - (const_logits[offset + target] - max_logit - log_sum_exp);
        losses[i] = loss;
    }
}

// Forward function for the CUDA module
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    // Check that tensors are CUDA tensors and have the expected dimensions and types
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be an Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    // Ensure the total number of elements fits in constant memory
    int total_elements = batch_size * num_classes;
    TORCH_CHECK(total_elements <= MAX_CONSTANT_ELEMENTS, "Predictions tensor is too large to fit in constant memory!");

    // Copy predictions into constant memory
    cudaError_t err = cudaMemcpyToSymbol(const_logits, predictions.data_ptr<float>(), total_elements * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    TORCH_CHECK(err == cudaSuccess, "Error copying predictions to constant memory: ", cudaGetErrorString(err));

    // Allocate output tensor for per-sample losses
    auto losses = torch::empty({batch_size}, predictions.options());

    // Launch the CUDA kernel
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    cross_entropy_loss_kernel_const<<<blocks, threads>>>(
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_const: ", cudaGetErrorString(err));

    // Compute mean loss over the batch
    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward using constant memory (CUDA)");
}
