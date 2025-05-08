#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define maximum constant memory sizes (in number of elements)
#define MAX_CONST_LOGITS_SIZE 4096  // Adjust this value if needed (e.g., supports up to 4096 floats, ~16KB)
#define MAX_CONST_TARGETS_SIZE 4096 // Adjust for number of targets

// Declare constant memory for read-only data
__constant__ float c_logits[MAX_CONST_LOGITS_SIZE];
__constant__ int64_t c_targets[MAX_CONST_TARGETS_SIZE];

// CUDA kernel that reads predictions and targets from constant memory
__global__ void cross_entropy_loss_kernel_const(
    float* losses,
    int batch_size,
    int num_classes
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size)
    {
        int base = i * num_classes;
        // Compute max logit for numerical stability
        float max_logit = c_logits[base];
        for (int j = 1; j < num_classes; j++) {
            max_logit = fmaxf(max_logit, c_logits[base + j]);
        }
        
        // Compute the sum of exp(logits - max_logit)
        float sum_exp = 0.0f;
        // Cache the shifted logit for the target class
        float target_shifted = c_logits[base + c_targets[i]] - max_logit;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += expf(c_logits[base + j] - max_logit);
        }
        
        float log_sum_exp = logf(sum_exp);
        // Compute cross entropy loss for the sample
        losses[i] = -(target_shifted - log_sum_exp);
    }
}


// Forward function for the PyTorch extension
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be an Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    TORCH_CHECK(targets.size(0) == batch_size, "Mismatch between predictions and targets batch sizes");
    TORCH_CHECK(batch_size * num_classes <= MAX_CONST_LOGITS_SIZE, "Predictions tensor size exceeds constant memory limit");
    TORCH_CHECK(batch_size <= MAX_CONST_TARGETS_SIZE, "Targets tensor size exceeds constant memory limit");

    // Copy predictions and targets to constant memory
    cudaMemcpyToSymbol(c_logits, predictions.data_ptr<float>(), batch_size * num_classes * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_targets, targets.data_ptr<int64_t>(), batch_size * sizeof(int64_t), 0, cudaMemcpyDeviceToDevice);

    // Allocate tensor for per-sample losses
    auto losses = torch::empty({batch_size}, predictions.options());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_loss_kernel_const<<<blocks, threads>>>(losses.data_ptr<float>(), batch_size, num_classes);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_const: ", cudaGetErrorString(err));

    // Compute and return mean loss over the batch
    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward with constant memory optimization (CUDA)");
}
